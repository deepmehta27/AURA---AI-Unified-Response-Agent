// API Request Types
export interface QueryRequest {
    query: string;
    use_rag?: boolean;
    history?: Array<{ role: string; content: string }>;
  }
  
  export interface UploadRequest {
    file: File;
    query?: string;
    task_type?: 'auto' | 'describe' | 'ocr' | 'transcribe' | 'summarize' | 'analyze';
  }
  
  // API Response Types
  export interface BaseResponse {
    success: boolean;
    error?: string;
    metadata?: Record<string, any>;
  }
  
  export interface AgentResponse extends BaseResponse {
    response: string;
    query_type?: string;
    agents_used?: string[];
    sources?: Array<{
      content: string;
      metadata: Record<string, any>;
    }>;
  }
  
  export interface HealthResponse {
    status: string;
    service: string;
  }
  
  // Agent Types
  export type AgentType = 'text' | 'image' | 'audio' | 'orchestrator';
  
  export type AnalysisType = 'describe' | 'ocr' | 'analyze' | 'question' | 'transcribe' | 'summarize';
  
  // UI State Types
  export interface QueryState {
    loading: boolean;
    error: string | null;
    response: AgentResponse | null;
  }
  