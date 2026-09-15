/**
 * Elysia MCP Plugin (MCP SDK v2)
 *
 * Auto-discovers routes and exposes them as MCP tools via a POST endpoint
 * handling the MCP JSON-RPC protocol. By default all routes are included;
 * opt out individual routes with `detail: { mcp: false }`, or set
 * `allRoutes: false` to require explicit `detail: { mcp: true }`.
 *
 * Uses `app.handle()` for tool invocation — every MCP tool call goes through
 * the full Elysia lifecycle (derive, resolve, beforeHandle, afterHandle, error
 * hooks, and all plugins).
 */

import { Elysia } from "elysia";
import type { DocumentDecoration, AnyElysia } from "elysia";

// ─── Module Augmentation ────────────────────────────────────────────
declare module "elysia" {
  interface DocumentDecoration {
    mcp?: boolean;
  }
}

import { McpServer } from "@modelcontextprotocol/server";
import type { Server } from "@modelcontextprotocol/server";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/types";

import { deriveToolName } from "./naming.js";
import { asSchemaLike, cleanResponseSchema, flattenSchemas, unflattenArgs } from "./schema.js";
import type { FlatJsonSchema, FlattenResult } from "./schema.js";
import { responseToMcpContent } from "./unwrap.js";

// ─── Types ───────────────────────────────────────────────────────────

export interface McpPluginOptions {
  /** MCP server name (default: "elysia-mcp") */
  name?: string;
  /** MCP server version (default: "1.0.0") */
  version?: string;
  /** Endpoint path (default: "/mcp") */
  path?: string;
  /** Expose all routes as MCP tools by default (default: true).
   *  When true, every route becomes a tool unless it sets `mcp: false`.
   *  When false, only routes with `detail: { mcp: true }` are exposed. */
  allRoutes?: boolean;
}

interface DiscoveredTool {
  name: string;
  description: string;
  method: string;
  pathSegments: string[];
  flatten: FlattenResult;
  outputSchema?: FlatJsonSchema;
}

type RouteHooks = {
  detail?: DocumentDecoration;
  params?: unknown;
  query?: unknown;
  body?: unknown;
  response?: unknown;
};

function discoverTools(
  app: Elysia,
  allRoutes: boolean,
  warn: (msg: string) => void,
): DiscoveredTool[] {
  const tools: DiscoveredTool[] = [];

  for (const route of app.routes) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const hooks = route.hooks as RouteHooks;
    const detail = hooks.detail;
    const mcpEnabled = detail?.mcp;

    if (mcpEnabled === false) continue;
    if (!allRoutes && (mcpEnabled === undefined || mcpEnabled === null)) continue;

    const method = route.method.toUpperCase();
    if (method === "OPTIONS" || method === "HEAD") continue;

    const routePath = route.path;
    const name = detail?.operationId ?? deriveToolName(method, routePath);
    const pathSegments = routePath.split("/");
    const description = detail?.summary ?? `${method} ${routePath}`;

    let flatten: FlattenResult;
    let outputSchema: FlatJsonSchema | undefined;

    try {
      const bodySchema = method === "GET" ? undefined : asSchemaLike(hooks.body);

      if (bodySchema !== undefined && bodySchema !== null && bodySchema["type"] !== "object") {
        warn(
          `[mcp] Tool "${name}": body schema type "${String(bodySchema["type"])}" cannot be represented as MCP tool arguments — route skipped`,
        );
        continue;
      }

      flatten = flattenSchemas(name, {
        params: asSchemaLike(hooks.params),
        query: asSchemaLike(hooks.query),
        body: bodySchema,
      });

      outputSchema = cleanResponseSchema(asSchemaLike(hooks.response));
    } catch (err) {
      warn(`[mcp] Route "${method} ${routePath}" skipped — schema conversion failed: ${String(err)}`);
      continue;
    }

    for (const warning of flatten.warnings) {
      warn(warning);
    }

    if (hooks.response !== undefined && hooks.response !== null && outputSchema === undefined) {
      warn(
        `[mcp] Tool "${name}": response schema is not type: "object" — outputSchema omitted`,
      );
    }

    tools.push({ name, description, method, pathSegments, flatten, outputSchema });
  }

  return tools;
}

// ─── Build synthetic Request for app.handle() ────────────────────────

function buildRequest(
  tool: DiscoveredTool,
  args: Record<string, unknown>,
  originalRequest: Request,
): Request {
  const { params, query, body } = unflattenArgs(args, tool.flatten);

  const resolvedPath = tool.pathSegments
    .map((segment) => {
      if (!segment.startsWith(":")) return segment;
      const key = segment.slice(1);
      if (!Object.prototype.hasOwnProperty.call(params, key)) {
        throw new McpError(
          ErrorCode.InvalidParams,
          `Missing required path parameter "${key}" for tool ${tool.name}`,
        );
      }
      return encodeURIComponent(String(params[key]));
    })
    .join("/");

  const queryEntries = Object.entries(query).filter(([, v]) => v !== null && v !== undefined);
  const qs =
    queryEntries.length > 0
      ? `?${new URLSearchParams(queryEntries.map(([k, v]) => [k, String(v)])).toString()}`
      : "";

  const origin = new URL(originalRequest.url).origin;
  const url = `${origin}${resolvedPath}${qs}`;

  const headers = new Headers(originalRequest.headers);
  headers.delete("content-length");
  if (body === undefined) {
    headers.delete("content-type");
  } else {
    headers.set("content-type", "application/json");
  }
  const bodyContent = body === undefined ? undefined : JSON.stringify(body);

  return new Request(url, {
    method: tool.method,
    headers,
    body: bodyContent,
  });
}

// ─── Create MCP Server with tool handlers (one per request) ─────────

function createMcpServer(
  serverName: string,
  serverVersion: string,
  toolMap: Map<string, DiscoveredTool>,
  toolListResponse: {
    tools: Array<{
      name: string;
      description: string;
      inputSchema: FlattenResult["schema"];
    }>;
  },
  rootApp: Elysia,
  originalRequest: Request,
): McpServer {
  const mcpServer = new McpServer({ name: serverName, version: serverVersion });

  // Use the underlying Server for custom request handlers.
  // We bypass registerTool() because our tools use pre-built JSON Schema
  // from flattenSchemas(), not Zod schemas.
  const server: Server = mcpServer.server;

  server.setRequestHandler(ListToolsRequestSchema, () => toolListResponse);

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const toolName = request.params.name;
    const tool = toolMap.get(toolName);

    if (!tool) {
      throw new McpError(ErrorCode.InvalidParams, `Tool ${toolName} not found`);
    }

    const args = request.params.arguments ?? {};

    // Build a synthetic request and run through the full Elysia lifecycle
    const syntheticRequest = buildRequest(tool, args, originalRequest);
    const response = await rootApp.handle(syntheticRequest);
    const result = await responseToMcpContent(response, tool.outputSchema !== undefined);

    if (!response.ok) {
      return {
        isError: true,
        content: result.content,
      };
    }

    // When the tool declares an outputSchema and the response is a JSON object,
    // include structuredContent so MCP clients can consume typed data.
    if (
      tool.outputSchema !== undefined &&
      result.parsed !== null &&
      result.parsed !== undefined &&
      typeof result.parsed === "object" &&
      !Array.isArray(result.parsed)
    ) {
      return {
        content: result.content,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        structuredContent: result.parsed as Record<string, unknown>,
      };
    }

    return { content: result.content };
  });

  return mcpServer;
}

// ─── Plugin ──────────────────────────────────────────────────────────

/**
 * Create the Elysia MCP plugin.
 *
 * By default all routes are exposed as MCP tools. Set `allRoutes: false`
 * to require explicit opt-in via `detail: { mcp: true }`, or opt out
 * individual routes with `detail: { mcp: false }`.
 */
export function mcp(options: McpPluginOptions = {}) {
  const { name = "elysia-mcp", version = "1.0.0", path = "/mcp", allRoutes = true } = options;

  return new Elysia({ name: "elysia-mcp" }).all(
    path,
    async ({ request }: { request: Request }) => {
      const emitted = new Set<string>();
      const warn = (msg: string): void => {
        if (emitted.has(msg)) return;
        emitted.add(msg);
        console.warn(msg);
      };

      // Discover tools from the parent app's routes via request context
      const rootApp = (globalThis as any).__ELYSIA_MCP_ROOT_APP__;
      if (!rootApp) {
        throw new Error("Elysia MCP: root app not registered. Call mcpRegisterRoot(app) first.");
      }

      let cachedRouteCount = -1;
      let toolMap = new Map<string, DiscoveredTool>();
      let toolListResponse: {
        tools: Array<{
          name: string;
          description: string;
          inputSchema: FlattenResult["schema"];
        }>;
      } = { tools: [] };

      function refreshTools() {
        if (rootApp.routes.length === cachedRouteCount) return;
        cachedRouteCount = rootApp.routes.length;

        const tools = discoverTools(rootApp, allRoutes, warn);

        toolMap = new Map<string, DiscoveredTool>();
        for (const tool of tools) {
          if (toolMap.has(tool.name)) {
            warn(`[mcp] Duplicate tool name "${tool.name}" — later route will override`);
          }
          toolMap.set(tool.name, tool);
        }
        toolListResponse = {
          tools: Array.from(toolMap.values(), (tool) => ({
            name: tool.name,
            description: tool.description,
            inputSchema: tool.flatten.schema,
            ...(tool.outputSchema === undefined ? {} : { outputSchema: tool.outputSchema }),
          })),
        };

        if (tools.length === 0) {
          warn("[mcp] No MCP-eligible routes found — MCP server will have no tools");
        }
      }

      refreshTools();

      // Create a fresh McpServer per request — the MCP SDK's
      // Protocol.connect() throws if the server is already connected,
      // so reusing a single instance across concurrent requests would fail.
      const server = createMcpServer(name, version, toolMap, toolListResponse, rootApp, request);

      // In v2, use handleRequest directly (no separate connect step)
      const response = await server.handleRequest(request);
      return response;
    },
    { detail: { mcp: false } }
  );
}

/**
 * Register the root Elysia app so the MCP plugin can discover its routes.
 * Must be called before any MCP requests are handled.
 */
export function mcpRegisterRoot(app: AnyElysia) {
  (globalThis as any).__ELYSIA_MCP_ROOT_APP__ = app;
}
