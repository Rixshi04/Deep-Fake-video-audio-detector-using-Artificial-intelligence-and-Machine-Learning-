import { toast } from "@/components/ui/use-toast";

// API base URL - make sure this matches your Flask server's address
const API_BASE_URL = 'http://192.168.1.39:5000';

export interface TaskResponse {
  task_id: string;
  status: string;
}

export interface TaskStatus {
  id: string;
  status: string;
  type: 'video' | 'audio';
  filename: string;
  message?: string;
  result?: {
    prediction: string;
    confidence: number;
    message?: string;
    spectrogram_url?: string;
    features?: Record<string, any>;
  };
  error?: string;
}

/**
 * Upload a video file for deepfake analysis
 */
export const uploadVideo = async (file: File, frames: number = 20): Promise<TaskResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('frames', frames.toString());

    const response = await fetch(`${API_BASE_URL}/api/upload/video`, {
      method: 'POST',
      body: formData,
      credentials: 'omit',
      mode: 'cors',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      // Try to parse error JSON if possible, otherwise use status text
      try {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server error: ${response.status} ${response.statusText}`);
      } catch (jsonError) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading video:', error);
    toast({
      title: "Upload Error",
      description: error instanceof Error ? error.message : "Failed to upload video",
      variant: "destructive",
    });
    throw error;
  }
};

/**
 * Upload an audio file for deepfake analysis
 */
export const uploadAudio = async (file: File): Promise<TaskResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/upload/audio`, {
      method: 'POST',
      body: formData,
      credentials: 'omit',
      mode: 'cors',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload audio');
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading audio:', error);
    toast({
      title: "Upload Error",
      description: error instanceof Error ? error.message : "Failed to upload audio",
      variant: "destructive",
    });
    throw error;
  }
};

/**
 * Check the status of a task
 */
export const checkTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`, {
      credentials: 'omit',
      mode: 'cors',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to check task status');
    }

    return await response.json();
  } catch (error) {
    console.error('Error checking task status:', error);
    throw error;
  }
};

/**
 * Poll a task until it's completed or fails
 */
export const pollTaskStatus = (
  taskId: string, 
  onProgress: (status: TaskStatus) => void,
  onComplete: (result: TaskStatus) => void,
  onError: (error: Error) => void
): (() => void) => {
  let isCancelled = false;
  
  const checkStatus = async () => {
    if (isCancelled) return;
    
    try {
      const status = await checkTaskStatus(taskId);
      
      if (status.status === 'completed') {
        onComplete(status);
        return;
      } else if (status.status === 'error') {
        onError(new Error(status.error || 'Task failed'));
        return;
      } else {
        onProgress(status);
        // Continue polling
        setTimeout(checkStatus, 1000);
      }
    } catch (error) {
      if (!isCancelled) {
        onError(error instanceof Error ? error : new Error('Failed to check task status'));
      }
    }
  };
  
  // Start polling
  checkStatus();
  
  // Return a function to cancel polling
  return () => {
    isCancelled = true;
  };
}; 