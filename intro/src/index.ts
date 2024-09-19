import dotenv from "dotenv";
dotenv.config();

import { tool } from "@langchain/core/tools";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";
import axios from "axios";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { MessageContent } from "@langchain/core/messages";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const getLocationInfo = tool(
  async (input: { locationId: string }) => {
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
      const response = await axios.request(configDetail);

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

const llm = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});

const webSearchTool = new TavilySearchResults({
  maxResults: 4,
});
const tools = [webSearchTool, getLocationInfo];

const toolNode = new ToolNode(tools); // A special node in the graph that calls tools when passed a tool call from the LLM.

// Call the model with the messages
//  MessagesAnnotation.State is a type that represents the state of the graph.
//  It is the way you define states in the graph
const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  const llmWithTools = llm.bindTools(tools);
  const result = await llmWithTools.invoke(messages);
  return { messages: [result] };
};

// This function checks if the last message was an AI message that called tools.
const shouldContinue = (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  // Extract the last message
  const lastMessage = messages[messages.length - 1];

  // Check if the last message was an AI message that called tools
  // If it was NOT, we should end the conversation - END
  // If it was, we should continue to the tools node - "tools"
  if (
    lastMessage._getType() !== "ai" ||
    !(lastMessage as AIMessage).tool_calls?.length
  ) {
    // LLM did not call any tools, or it's not an AI message, so we should end.
    return END;
  }
  return "tools";
};

/**
 * MessagesAnnotation is a pre-built state annotation imported from @langchain/langgraph.
 * It is the same as the following annotation:
 *
 * - This defines the object that is passed between each node
 *   in the graph. We will create different nodes for each agent and tool
 * ```typescript
 * const MessagesAnnotation = Annotation.Root({
 *   messages: Annotation<BaseMessage[]>({
 *     reducer: messagesStateReducer,
 *     default: () => [systemMessage],
 *   }),
 * // The agent node that last performed work
  next: Annotation<string>({
    reducer: (x, y) => y ?? x ?? END,
    default: () => END,
  }),
 * });
 * ```
 */

// Define the workflow
// - Pass the MessagesAnnotation to the StateGraph constructor
// - Add the callModel node to the graph
// - Add the toolNode to the graph
// - Add edges to the graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel) // This is the agent node
  .addEdge(START, "agent")
  .addNode("tools", toolNode) // This is the tools node
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END]);

export const graph = workflow.compile({
  // The LangGraph Studio/Cloud API will automatically add a checkpointer
  // only uncomment if running locally
  // checkpointer: new MemorySaver(),
});

// let inputs = {
//   messages: [{ role: "user", content: "What is the hotel's exact address?" }],
// };
// let config = {
//   configurable: { locationId: "229968", user_id: "example_user" },
// };

// (async () => {
//   for await (const chunk of await graph.stream(inputs, config)) {
//     for (const [node, values] of Object.entries(chunk)) {
//       if (node === "agent") {
//         const message = (values as { messages: AIMessage[] }).messages[0];
//         if (message instanceof AIMessage) {
//           console.log(message.content);
//         }
//       }
//     }
//   }
// })();
