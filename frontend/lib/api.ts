import axios, { AxiosInstance, AxiosError, AxiosResponse } from 'axios';
import { QueryRequest, AgentResponse, HealthResponse } from './types';

class APIClient {
  private client: AxiosInstance;
  
  constructor(baseURL: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 60000, // 60 seconds for AI processing
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/api/health');
    return response.data;
  }

  // Text Agent - Query endpoint
  async queryText(request: QueryRequest): Promise<AgentResponse> {
    const response = await this.client.post<AgentResponse>(
      '/api/query',
      request
    );
    return response.data;
  }

  // Image Agent - Upload and analyze
  async analyzeImage(
    file: File,
    query?: string,
    taskType: string = 'auto'
  ): Promise<AgentResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (query) {
      formData.append('query', query);
    }
    formData.append('task_type', taskType);

    const response = await this.client.post<AgentResponse>(
      '/api/upload/analyze',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  // Audio Agent - Upload and process
  async processAudio(
    file: File,
    analysisType: string = 'transcribe',
    query?: string
  ): Promise<AgentResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('analysis_type', analysisType);
    
    if (query) {
      formData.append('query', query);
    }

    const response = await this.client.post<AgentResponse>(
      '/api/upload/analyze',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  // Orchestrator - Multi-modal processing
  async processMultiModal(
    query: string,
    file?: File,
    useRag: boolean = true
  ): Promise<AgentResponse> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('use_rag', useRag.toString());
    
    if (file) {
      formData.append('file', file);
    }

    const response = await this.client.post<AgentResponse>(
      '/api/orchestrator/process',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export class for custom instances
export default APIClient;
