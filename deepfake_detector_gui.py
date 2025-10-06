import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk

# Import our detector functions
from simple_deepfake_detector import predict_deepfake

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Set app title
        title_frame = tk.Frame(root, bg="#f0f0f0")
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame, 
            text="Deepfake Detection System",
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame, 
            text="Detect manipulated videos using deep learning",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#666666"
        )
        subtitle_label.pack(pady=5)
        
        # Main content area
        content_frame = tk.Frame(root, bg="#f0f0f0")
        content_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Video preview area
        self.preview_frame = tk.Frame(content_frame, bg="#e0e0e0", height=300)
        self.preview_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        self.preview_label = tk.Label(
            self.preview_frame, 
            text="Video Preview",
            bg="#e0e0e0"
        )
        self.preview_label.pack(pady=100)
        
        # Control panel
        control_frame = tk.Frame(root, bg="#f0f0f0")
        control_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # File selection
        file_frame = tk.Frame(control_frame, bg="#f0f0f0")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_path_entry = tk.Entry(
            file_frame, 
            textvariable=self.file_path_var,
            width=50,
            font=("Arial", 10)
        )
        file_path_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        browse_button = tk.Button(
            file_frame, 
            text="Browse",
            command=self.browse_file,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15
        )
        browse_button.pack(side=tk.RIGHT)
        
        # Sequence length control
        seq_frame = tk.Frame(control_frame, bg="#f0f0f0")
        seq_frame.pack(fill=tk.X, pady=5)
        
        seq_label = tk.Label(
            seq_frame, 
            text="Frames to analyze:",
            bg="#f0f0f0",
            font=("Arial", 10)
        )
        seq_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.seq_length_var = tk.IntVar(value=20)
        seq_slider = tk.Scale(
            seq_frame,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seq_length_var,
            bg="#f0f0f0",
            length=300
        )
        seq_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Analyze button
        analyze_button = tk.Button(
            control_frame, 
            text="ANALYZE VIDEO",
            command=self.analyze_video,
            bg="#007BFF",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        analyze_button.pack(pady=15, fill=tk.X)
        
        # Results area
        self.result_frame = tk.Frame(root, bg="#f0f0f0")
        self.result_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Create a static label in result frame that won't be destroyed
        self.status_label = tk.Label(
            self.result_frame, 
            text="Upload a video to analyze",
            font=("Arial", 14),
            bg="#f0f0f0"
        )
        self.status_label.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            root, 
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            root, 
            orient=tk.HORIZONTAL,
            length=800, 
            mode='indeterminate'
        )
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Video file path
        self.video_path = None
        self.cap = None
        
        # Flag to track if analysis is running
        self.analysis_running = False
        
    def browse_file(self):
        """Open a file dialog to select a video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.video_path = file_path
            self.file_path_var.set(file_path)
            self.load_video_preview()
        
    def load_video_preview(self):
        """Load the first frame of the video as preview"""
        if self.video_path and os.path.exists(self.video_path):
            # Clear previous preview
            for widget in self.preview_frame.winfo_children():
                widget.destroy()
                
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert frame to preview image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (480, 320))
                    img = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    # Display preview
                    preview_img = tk.Label(self.preview_frame, image=img_tk, bg="#e0e0e0")
                    preview_img.image = img_tk  # Keep a reference
                    preview_img.pack(padx=10, pady=10)
                    
                    # Display video info
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    info_text = f"Duration: {duration:.2f}s | Frames: {frame_count} | FPS: {fps:.2f}"
                    info_label = tk.Label(
                        self.preview_frame, 
                        text=info_text,
                        bg="#e0e0e0",
                        font=("Arial", 9)
                    )
                    info_label.pack()
                    
                else:
                    self.preview_label = tk.Label(
                        self.preview_frame, 
                        text="Could not read video frame",
                        bg="#e0e0e0"
                    )
                    self.preview_label.pack(pady=100)
                
                cap.release()
            else:
                self.preview_label = tk.Label(
                    self.preview_frame, 
                    text="Could not open video file",
                    bg="#e0e0e0"
                )
                self.preview_label.pack(pady=100)
    
    def analyze_video(self):
        """Analyze the selected video in a separate thread"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a valid video file first")
            return
            
        if self.analysis_running:
            messagebox.showinfo("Info", "Analysis is already running")
            return
            
        # Clear previous results by updating status label
        self.status_label.config(text="Analyzing... Please wait.")
        
        # Update UI
        self.status_var.set("Analyzing video...")
        self.progress.start(10)
        self.analysis_running = True
        
        # Start analysis in a separate thread
        threading.Thread(target=self._run_analysis, daemon=True).start()
    
    def _run_analysis(self):
        """Run the analysis in a background thread"""
        try:
            # Get the sequence length
            seq_length = self.seq_length_var.get()
            
            # Run prediction
            result = predict_deepfake(self.video_path, seq_length)
            
            # Update UI with results
            if 'error' in result:
                self.root.after(0, lambda: self._show_error(result['error']))
            else:
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Update UI from main thread
                self.root.after(0, lambda: self._show_result(prediction, confidence))
                
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            # Stop progress animation from main thread
            self.root.after(0, self._analysis_complete)
    
    def _show_result(self, prediction, confidence):
        """Display the analysis result"""
        # Clear previous widgets except the status label
        for widget in self.result_frame.winfo_children():
            if widget != self.status_label:
                widget.destroy()
        
        # Update status label
        self.status_label.config(
            text=f"Prediction: {prediction} (Confidence: {confidence:.2f}%)",
            font=("Arial", 16, "bold"),
            fg="#4CAF50" if prediction == "REAL" else "#F44336"
        )
    
    def _show_error(self, error_message):
        """Display an error message"""
        # Update status label with error
        self.status_label.config(
            text=f"Error: {error_message}",
            font=("Arial", 14),
            fg="#F44336"
        )
    
    def _analysis_complete(self):
        """Update UI after analysis is complete"""
        self.progress.stop()
        self.status_var.set("Analysis complete")
        self.analysis_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop() 