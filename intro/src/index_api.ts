import "dotenv/config";

import { tool } from "@langchain/core/tools";

import axios from "axios";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
  Annotation,
} from "@langchain/langgraph";
import { MessageContent } from "@langchain/core/messages";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

// Create Agent Supervisor
import { z } from "zod";
import { JsonOutputToolsParser } from "langchain/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

// This defines the object that is passed between each node
// in the graph. We will create different nodes for each agent and tool
const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  // The agent node that last performed work
  next: Annotation<string>({
    reducer: (x, y) => y ?? x ?? END,
    default: () => END,
  }),
});

// Define the tool to search for location info
const getLocationInfo = tool(
  async (input: { locationId: string }) => {
    console.log("getLocationInfo tool called");
    let locationId = input.locationId;

    let configDetail = {
      method: "get",
      maxBodyLength: Infinity,
      url: `http://api.tripadvisor.com/api/partner/2.0/location/${locationId}`,
      headers: {
        "X-TripAdvisor-API-Key": process.env.TRIPADVISOR_API_KEY,
        "User-Agent": "cashew",
      },
    };

    try {
      let response = await axios.request(configDetail);
      return response.data;
    } catch (error) {
      console.log(error);
      return {};
    }
  },
  {
    name: "get_location_info",
    description:
      "Call the tripadvisor api to get location info including address, location ratings, user reviews, awards, location name, user rating count.",
    schema: z.object({
      locationId: z
        .string()
        .describe("Location id to search the trip advisor api"),
    }),
  }
);

// Define the Tavily tool
const tavilyTool = new TavilySearchResults();

// Define the roles of the members
const members = ["researcher", "tripadvisorApi"] as const;

// Define the system prompt
const systemPrompt =
  "You are a supervisor tasked with managing a conversation between the" +
  " following workers: {members}. Given the following user request," +
  " respond with the worker to act next. Each worker will perform a" +
  " task and respond with their results and status. When finished," +
  " respond with FINISH.";

// Define the user prompt
const options = [END, ...members];

// Define the routing function
const routingTool = {
  name: "route",
  description: "Select the next role.",
  schema: z.object({
    next: z.enum([END, ...members]),
  }),
};

// Define the prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemPrompt],
  new MessagesPlaceholder("messages"),
  [
    "system",
    "Given the conversation above, who should act next?" +
      " Or should we FINISH? Select one of: {options}",
  ],
]);

// Format the prompt
const formattedPrompt = await prompt.partial({
  options: options.join(", "),
  members: members.join(", "),
});

// Define the ChatOpenAI instance
const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0,
});

// Define the supervisor chain
const supervisorChain = formattedPrompt
  .pipe(
    llm.bindTools([routingTool], {
      tool_choice: "route",
    })
  )
  .pipe(new JsonOutputToolsParser())
  // select the first one
  .pipe((x) => x[0].args);

// Import the HumanMessage class
import { HumanMessage } from "@langchain/core/messages";

// Call the supervisor chain
await supervisorChain.invoke({
  messages: [
    new HumanMessage({
      content: "What is the hotel's exact address? locationId: 229968",
    }),
  ],
});

// Construct the graph
import { RunnableConfig } from "@langchain/core/runnables";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { SystemMessage } from "@langchain/core/messages";
import { get } from "@langchain/community/utils/convex";

// Define the researcher agent
const researcherAgent = createReactAgent({
  llm,
  tools: [tavilyTool],
  messageModifier: new SystemMessage(
    "You are a web researcher. You may use the Tavily search engine to search the web for" +
      " important information, so the Chart Generator in your team can make useful plots."
  ),
});

// Define the researcher node
const researcherNode = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const result = await researcherAgent.invoke(state, config);
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [
      new HumanMessage({ content: lastMessage.content, name: "Researcher" }),
    ],
  };
};

// Define the tripadvisor api agent
const tripadvisorApiAgent = createReactAgent({
  llm,
  tools: [getLocationInfo],
  messageModifier: new SystemMessage(
    "You excel at getting location information such as address, location ratings, user reviews, awards, location name, user rating count."
  ),
});

const tripadvisorApiNode = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const result = await tripadvisorApiAgent.invoke(state, config);
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [
      new HumanMessage({
        content: lastMessage.content,
        name: "TripadvisorAPI",
      }),
    ],
  };
};

// Now we can create the graph itself! Add the nodes, and add edges to define how how work will be performed in the graph.
// import { START, StateGraph } from "@langchain/langgraph";

// 1. Create the graph
const workflow = new StateGraph(AgentState)
  // 2. Add the nodes; these will do the work
  .addNode("researcher", researcherNode)
  .addNode("tripadvisorApi", tripadvisorApiNode)
  .addNode("supervisor", supervisorChain);
// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
members.forEach((member) => {
  workflow.addEdge(member as "researcher" | "tripadvisorApi", "supervisor");
});

workflow.addConditionalEdges(
  "supervisor",
  (x: typeof AgentState.State) => x.next
);

workflow.addEdge(START, "supervisor");

const graph = workflow.compile();

// Now we can invoke the graph with the initial state
let streamResults = graph.stream(
  {
    messages: [
      new HumanMessage({
        content: "What is the exact address? locationId: 229968",
      }),
    ],
  },
  { recursionLimit: 100 }
);

for await (const output of await streamResults) {
  if (!output?.__end__) {
    console.log(output);
    console.log("----");
  }
}

// -------------------------------------------------------------------------------

// const getLocationInfo = tool(
//   async (input: { locationId: string }) => {
//     let locationId = input.locationId;

//     let configDetail = {
//       method: "get",
//       maxBodyLength: Infinity,
//       url: `http://api.tripadvisor.com/api/partner/2.0/location/${locationId}`,
//       headers: {
//         "X-TripAdvisor-API-Key": process.env.TRIPADVISOR_API_KEY,
//         "User-Agent": "cashew",
//       },
//     };

//     try {
//       const response = await axios.request(configDetail);

//       return response.data;
//     } catch (error) {
//       console.log(error);
//       return {};
//     }
//   },
//   {
//     name: "get_location_info",
//     description:
//       "Call the tripadvisor api to get location info including address, location ratings, user reviews, awards, location name, user rating count.",
//     schema: z.object({
//       locationId: z
//         .string()
//         .describe("Location id to search the trip advisor api"),
//     }),
//   }
// );

// const llm = new ChatOpenAI({
//   model: "gpt-4o",
//   temperature: 0,
//   apiKey: process.env.OPENAI_API_KEY,
// });

// const webSearchTool = new TavilySearchResults({
//   maxResults: 4,
// });
// const tools = [webSearchTool, getLocationInfo];

// const toolNode = new ToolNode(tools); // A special node in the graph that calls tools when passed a tool call from the LLM.

// // Call the model with the messages
// //  MessagesAnnotation.State is a type that represents the state of the graph.
// //  It is the way you define states in the graph
// const callModel = async (state: typeof MessagesAnnotation.State) => {
//   const { messages } = state;

//   const llmWithTools = llm.bindTools(tools);
//   const result = await llmWithTools.invoke(messages);
//   return { messages: [result] };
// };

// // This function checks if the last message was an AI message that called tools.
// const shouldContinue = (state: typeof MessagesAnnotation.State) => {
//   const { messages } = state;

//   // Extract the last message
//   const lastMessage = messages[messages.length - 1];

//   // Check if the last message was an AI message that called tools
//   // If it was NOT, we should end the conversation - END
//   // If it was, we should continue to the tools node - "tools"
//   if (
//     lastMessage._getType() !== "ai" ||
//     !(lastMessage as AIMessage).tool_calls?.length
//   ) {
//     // LLM did not call any tools, or it's not an AI message, so we should end.
//     return END;
//   }
//   return "tools";
// };

// /**
//  * MessagesAnnotation is a pre-built state annotation imported from @langchain/langgraph.
//  * It is the same as the following annotation:
//  *
//  * - This defines the object that is passed between each node
//  *   in the graph. We will create different nodes for each agent and tool
//  * ```typescript
//  * const MessagesAnnotation = Annotation.Root({
//  *   messages: Annotation<BaseMessage[]>({
//  *     reducer: messagesStateReducer,
//  *     default: () => [systemMessage],
//  *   }),
//  * // The agent node that last performed work
//   next: Annotation<string>({
//     reducer: (x, y) => y ?? x ?? END,
//     default: () => END,
//   }),
//  * });
//  * ```
//  */

// // Define the workflow
// // - Pass the MessagesAnnotation to the StateGraph constructor
// // - Add the callModel node to the graph
// // - Add the toolNode to the graph
// // - Add edges to the graph
// const workflow = new StateGraph(MessagesAnnotation)
//   .addNode("agent", callModel) // This is the agent node
//   .addEdge(START, "agent")
//   .addNode("tools", toolNode) // This is the tools node
//   .addEdge("tools", "agent")
//   .addConditionalEdges("agent", shouldContinue, ["tools", END]);

// export const graph = workflow.compile({
//   // The LangGraph Studio/Cloud API will automatically add a checkpointer
//   // only uncomment if running locally
//   // checkpointer: new MemorySaver(),
// });

// // let inputs = {
// //   messages: [{ role: "user", content: "What is the hotel's exact address?" }],
// // };
// // let config = {
// //   configurable: { locationId: "229968", user_id: "example_user" },
// // };

// // (async () => {
// //   for await (const chunk of await graph.stream(inputs, config)) {
// //     for (const [node, values] of Object.entries(chunk)) {
// //       if (node === "agent") {
// //         const message = (values as { messages: AIMessage[] }).messages[0];
// //         if (message instanceof AIMessage) {
// //           console.log(message.content);
// //         }
// //       }
// //     }
// //   }
// // })();
