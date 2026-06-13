export type ConnectionState = "connecting" | "connected" | "reconnecting" | "closed";

export type MonitorEventName =
  | "session_created"
  | "tool_start"
  | "assistant_call"
  | "task_result"
  | "task_cancelled"
  | "error"
  | string;

export interface MonitorMessage {
  type: "monitor_event";
  event: MonitorEventName;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
  trace_id?: string | null;
  thread_id?: string | null;
  span_id?: string | null;
}

export interface PongMessage {
  type: "pong";
  message: string;
}

export type SocketMessage = MonitorMessage | PongMessage;

export interface TaskResponse {
  status: "started" | string;
  thread_id: string;
  trace_id?: string;
}

export interface CancelTaskResponse {
  status: "cancelled" | "cancelling" | string;
  thread_id: string;
  message?: string;
}

export interface UploadResponse {
  status: "uploaded" | string;
  files: string[];
}

export interface OutputFile {
  name: string;
  type: "file" | string;
  path: string;
  size: number;
  mtime: number;
}

export interface FileListResponse {
  files?: OutputFile[];
  error?: string;
}

export interface UploadedItem {
  uid: string;
  name: string;
  size: number;
  raw: File;
}

export type MemoryCategory = "profile" | "preference" | "project";

export interface MemoryRecord {
  id: string;
  category: MemoryCategory;
  key: string;
  content: string;
  confidence: number;
  source_thread_id: string;
  source_trace_id: string;
  created_at: string;
  updated_at: string;
}

export interface MemoryListResponse {
  memories: MemoryRecord[];
  count: number;
}

export interface MemoryDeleteResponse {
  status: "deleted" | string;
  deleted_ids: string[];
}
