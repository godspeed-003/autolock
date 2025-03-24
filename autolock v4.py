import os
import cv2
import time
import argparse
import numpy as np
import pickle
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
import threading
import ctypes
import platform

class FaceRecognitionSystem:
    def __init__(self, model_path='yolov8n-face.pt', data_dir='face_data', 
                 confidence_threshold=0.5, master_name="master",
                 inactivity_timeout=5, check_interval=1, 
                 recognition_tolerance=0.4):
        # Initialize parameters
        self.model_path = model_path
        self.data_dir = data_dir
        self.confidence_threshold = confidence_threshold
        self.master_name = master_name
        self.inactivity_timeout = 5
        self.check_interval = check_interval
        
        # Create directory for face data if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        self.blacklist_file = os.path.join(self.data_dir, "blacklist.pkl")
        self.blacklist = self.load_blacklist()

        # Load YOLOv8 model
        print("Loading YOLOv8 face detection model...")
        self.model = YOLO(self.model_path)
        
        # Initialize face recognition system
        self.face_recognizer = None
        self.embeddings_file = os.path.join(self.data_dir, f"{self.master_name}_embeddings.pkl")
        self.knn_model_file = os.path.join(self.data_dir, f"{self.master_name}_knn_model.pkl")
        
        # Load face recognition model if it exists
        self.load_face_recognition_model()
        
        # Initialize webcam
        self.cap = None
        
        # System state
        self.master_present = False
        self.last_activity_time = time.time()
        self.running = True
        self.recognition_tolerance = recognition_tolerance
        self.recent_confidences = []
        self.confidence_history_size = 10
        
        # System lock control thread
        self.lock_control_thread = threading.Thread(target=self.monitor_presence)
    
    def load_blacklist(self):
        """Load blacklisted embeddings from file"""
        if os.path.exists(self.blacklist_file):
            with open(self.blacklist_file, 'rb') as f:
                return pickle.load(f)
        return []

    def add_to_blacklist(self, face_img):
        """Extract embedding and add it to blacklist"""
        embedding = self.extract_features_with_dim_check(face_img)
        self.blacklist.append(embedding)

        # Save blacklist to file
        with open(self.blacklist_file, 'wb') as f:
            pickle.dump(self.blacklist, f)

    def load_face_recognition_model(self):
        """Load the KNN model for face recognition if it exists"""
        if os.path.exists(self.knn_model_file):
            print(f"Loading existing face recognition model for {self.master_name}...")
            with open(self.knn_model_file, 'rb') as f:
                self.face_recognizer = pickle.load(f)
            return True
        return False
    
    def create_master_profile(self):
        """Create a profile for the master user by collecting face embeddings until spacebar is pressed"""
        print(f"Creating profile for {self.master_name}...")
        print("Collecting face samples. Please move your head slowly in different angles.")
        print("Tip: Try to capture your face in different lighting conditions and distances.")
        print("Press SPACEBAR when you think enough samples have been collected.")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Fallback to camera 0
            if not self.cap.isOpened():
                raise Exception("Could not open any webcam")
        
        # Collect face embeddings
        face_embeddings = []
        sample_count = 0
        continue_training = True
        last_sample_time = time.time()
        min_sample_interval = 0.2  # Minimum time between samples (seconds)
        
        # For validation during training
        validation_interval = 10  # Validate after every 10 samples
        recognition_accuracy = 0.0
        current_model = None
        
        while continue_training:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            elapsed = current_time - last_sample_time
            
            # Detect faces
            results = self.model(frame)
            
            # Display frame
            display_frame = frame.copy()
            
            # Check if we need to train temporary model for validation
            if len(face_embeddings) > 0 and sample_count % validation_interval == 0 and current_model is None:
                embeddings_array = np.array(face_embeddings)
                labels = np.array([f"{self.master_name}_{i}" for i in range(len(face_embeddings))])
                
                # Train temporary KNN model
                current_model = KNeighborsClassifier(n_neighbors=min(3, len(face_embeddings)))
                current_model.fit(embeddings_array, labels)
            
            # Process results
            if len(results[0].boxes) > 0:
                # Get the face with highest confidence
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                if confidences[best_idx] > self.confidence_threshold:
                    # Extract face embedding
                    bbox = boxes[best_idx].xyxy.cpu().numpy()[0]
                    face_crop = self.get_face_crop(frame, bbox)
                    
                    if face_crop is not None:
                        # Draw rectangle around detected face
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Resize to fixed size for consistency
                        face_resized = cv2.resize(face_crop, (112, 112))
                        
                        # Test current recognition if we have a model
                        if current_model is not None:
                            # Extract features
                            embedding = self.extract_features(face_resized)
                            
                            # Get prediction and confidence
                            try:
                                distances, _ = current_model.kneighbors([embedding], n_neighbors=min(3, len(face_embeddings)))
                                avg_distance = np.mean(distances[0])
                                max_distance = 20.0
                                recognition_accuracy = max(0, 1 - (avg_distance / max_distance)) * 100
                            except Exception as e:
                                print(f"Validation error: {e}")
                                recognition_accuracy = 0.0
                        
                        # Add new sample at defined interval
                        if elapsed >= min_sample_interval:
                            # Extract features for training
                            embedding = self.extract_features(face_resized)
                            
                            # For the first sample, save feature dimension
                            if sample_count == 0:
                                feature_dim_file = os.path.join(self.data_dir, f"{self.master_name}_feature_dim.txt")
                                with open(feature_dim_file, 'w') as f:
                                    f.write(str(len(embedding)))
                                print(f"Saved feature dimension: {len(embedding)}")
                            
                            face_embeddings.append(embedding)
                            sample_count += 1
                            last_sample_time = current_time
                            
                            # Reset validation model after collecting more samples
                            if sample_count % validation_interval == 0:
                                current_model = None
            
            # Display training status
            cv2.putText(display_frame, f"Samples collected: {sample_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display recognition accuracy if available
            if recognition_accuracy > 0:
                cv2.putText(display_frame, f"Recognition accuracy: {recognition_accuracy:.1f}%", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Press SPACEBAR when satisfied with training", 
                    (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Face Collection", display_frame)
            
            # Handle user input
            key = cv2.waitKey(1)
            if key == 32:  # SPACEBAR
                print(f"Training stopped by user after collecting {sample_count} samples")
                continue_training = False
            elif key == 27:  # ESC key
                print("Training canceled by user")
                cv2.destroyAllWindows()
                self.cap.release()
                return False
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        if sample_count < 20:
            print(f"Warning: Only collected {sample_count} samples. This might not be enough for reliable recognition.")
            proceed = input("Do you want to proceed with the limited samples? (y/n): ").lower()
            if proceed != 'y':
                return False
        
        # Train KNN model with collected embeddings
        if len(face_embeddings) > 0:
            embeddings_array = np.array(face_embeddings)
            labels = np.array([f"{self.master_name}_{i}" for i in range(len(face_embeddings))])
            print("Unique labels before training:", np.unique(labels))

            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump((embeddings_array, labels), f)

            # Ensure k is valid (minimum 1, max 10% of dataset)
            k_neighbors = min(5, max(1, len(face_embeddings) // 10))  
            print(f"Training KNN with {len(face_embeddings)} samples, k={k_neighbors}")

            self.face_recognizer = KNeighborsClassifier(n_neighbors=k_neighbors)
            self.face_recognizer.fit(embeddings_array, labels)

            # Debug: Print number of unique classes
            print(f"KNN trained with {len(self.face_recognizer.classes_)} unique classes")

            # Save KNN model
            with open(self.knn_model_file, 'wb') as f:
                pickle.dump(self.face_recognizer, f)

            print(f"Successfully created and saved profile for {self.master_name} with {len(face_embeddings)} samples")
            return True

        print("Failed to create profile: Not enough face samples collected")
        return False
    
    def get_face_crop(self, frame, bbox):
        """Crop face from frame using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        # Add some margin
        margin = 20  # Increased margin for better feature extraction
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return face_crop
    
    def extract_features(self, face_img):
        """Extract feature vector from face image using HOG features"""
        # Convert to grayscale
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to smaller fixed size to reduce feature vector length
        face_img_gray = cv2.resize(face_img_gray, (64, 64))  

        # Apply histogram equalization
        face_img_eq = cv2.equalizeHist(face_img_gray)

        # Extract HOG features with reduced parameters
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 6  # Reduced from 9 to lower dimensionality

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(face_img_eq)

        # Downsize pixel intensity features and normalize
        downsized = cv2.resize(face_img_eq, (8, 8)).flatten() / 255.0

        # Combine HOG and pixel features
        combined_features = np.concatenate([hog_features.flatten(), downsized])

        return combined_features
    
    def get_feature_dimensions(self):
        """Get the expected feature dimensions from the trained model"""
        if self.face_recognizer is None:
            return None
        # For KNeighborsClassifier, the feature dimension is in the shape of X
        if hasattr(self.face_recognizer, '_fit_X'):
            return self.face_recognizer._fit_X.shape[1]
        return None

    def extract_features_with_dim_check(self, face_img):
        """Extract features and ensure they match the expected dimensions"""
        # Get current feature vector
        embedding = self.extract_features(face_img)
        
        # Check if we need to match dimensions
        expected_dim = self.get_feature_dimensions()
        if expected_dim is not None and len(embedding) != expected_dim:
            print(f"WARNING: Feature dimension mismatch. Got {len(embedding)}, expected {expected_dim}")
            
            # Fix by padding or truncating
            if len(embedding) < expected_dim:
                embedding = np.pad(embedding, (0, expected_dim - len(embedding)), 'constant')
            else:
                embedding = embedding[:expected_dim]
        
        return embedding

    def recognize_face(self, face_img):
        """Recognize a face using KNN and apply confidence threshold"""
        if self.face_recognizer is None:
            return None, 0.0

        embedding = self.extract_features_with_dim_check(face_img)
        available_neighbors = len(self.face_recognizer.classes_)

        if available_neighbors < 2:
            print("ERROR: Not enough classes in KNN model!")
            return None, 0.0

        k = min(5, available_neighbors)

        try:
            distances, indices = self.face_recognizer.kneighbors([embedding], n_neighbors=k)
            predicted_class = self.face_recognizer.classes_[indices[0][0]]
            avg_distance = np.mean(distances[0])
            max_distance = 20.0
            confidence = max(0, 1 - (avg_distance / max_distance)) * 100  

            # If this embedding is in the blacklist, force it to "Unknown"
            for blacklisted_embedding in self.blacklist:
                if np.linalg.norm(blacklisted_embedding - embedding) < 3.0:  # Similar to KNN threshold
                    print("⚠️ Blacklisted face detected! Marking as unknown.")
                    return None, 0.0

            # If confidence is above 75%, return as master
            if confidence >= 75:
                return self.master_name, confidence

            return None, confidence

        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 0.0
    
    def prevent_sleep(self):
        """Prevent the system from sleeping"""
        if platform.system() == 'Windows':
            ctypes.windll.kernel32.SetThreadExecutionState(
                0x80000002  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
        elif platform.system() == 'Darwin':  # macOS
            os.system('caffeinate -u -t 60')  # Keep display on for 60 seconds
        elif platform.system() == 'Linux':
            os.system('xdg-screensaver reset')
    
    def lock_system(self):
        """Lock the system"""
        if platform.system() == 'Windows':
            ctypes.windll.user32.LockWorkStation()
        elif platform.system() == 'Darwin':  # macOS
            os.system('pmset displaysleepnow')
        elif platform.system() == 'Linux':
            os.system('xdg-screensaver lock')
    
    def allow_sleep(self):
        """Allow the system to sleep normally"""
        if platform.system() == 'Windows':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
    
    def monitor_presence(self):
        """Thread function to monitor master presence and control system lock"""
        while self.running:
            current_time = time.time()
            
            # If master was present but now absent for too long, lock system
            if not self.master_present and (current_time - self.last_activity_time) > self.inactivity_timeout:
                print("Master absent - locking system")
                self.lock_system()
                # Reset timer to avoid multiple locks
                self.last_activity_time = current_time
                self.unlock_system()  # Start monitoring for unlocking
            
            # Sleep to avoid excessive CPU usage
            time.sleep(self.check_interval)
    
    def unlock_system(self):
        """Unlocks Windows when the master is detected after locking."""
        pin = "2486"  # Store your 4-digit PIN here

        print("Waiting for master to return to unlock system...")

        while True:
            cap = cv2.VideoCapture(0)  # Open webcam
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1)
                continue

            results = self.model(frame)
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    bbox = box.xyxy.cpu().numpy()[0]
                    confidence = box.conf.cpu().numpy()[0]

                    if confidence > self.confidence_threshold:
                        face_crop = self.get_face_crop(frame, bbox)
                        if face_crop is not None:
                            face_resized = cv2.resize(face_crop, (112, 112))
                            identity, rec_confidence = self.recognize_face(face_resized)

                            if identity == self.master_name and rec_confidence >= 75:
                                print("Master detected - unlocking system")

                                if platform.system() == 'Windows':
                                    time.sleep(2)  # Ensure Windows login screen is ready
                                    ctypes.windll.user32.LockWorkStation()  # Bring up login screen
                                    time.sleep(2)
                                    import pyautogui
                                    pyautogui.typewrite(pin)  # Enter stored PIN
                                    pyautogui.press('enter')

                                cap.release()
                                return  # Exit once unlocked

            cap.release()
            time.sleep(2)  # Avoid excessive CPU usage

    def run(self):
        """Main run loop for the face recognition system"""
        # Check if we need to create a master profile
        if not self.load_face_recognition_model():
            create_profile = input("No master profile found. Create one? (y/n): ").lower()
            if create_profile == 'y':
                success = self.create_master_profile()
                if not success:
                    print("Failed to create master profile. Exiting.")
                    return
            else:
                print("Cannot continue without master profile. Exiting.")
                return
        
        # Start the lock control thread
        self.lock_control_thread.daemon = True
        self.lock_control_thread.start()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Fallback to camera 0
            if not self.cap.isOpened():
                raise Exception("Could not open any webcam")
        
        print("Face recognition system active. Press 'q' to quit.")
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Detect faces using YOLOv8
                results = self.model(frame)
                
                # Process detection results
                display_frame = frame.copy()
                self.master_present = False
                
                if len(results[0].boxes) > 0:
                    # Process each detected face
                    for box in results[0].boxes:
                        bbox = box.xyxy.cpu().numpy()[0]
                        confidence = box.conf.cpu().numpy()[0]
                        
                        if confidence > self.confidence_threshold:
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Crop and recognize face
                            face_crop = self.get_face_crop(frame, bbox)
                            if face_crop is not None:
                                face_resized = cv2.resize(face_crop, (112, 112))
                                identity, rec_confidence = self.recognize_face(face_resized)
                                
                                # Display recognition results with percentage match
                                match_percentage = rec_confidence * 100
                                color = (0, 0, 255)  # Default red for unknown
                                
                                if identity == self.master_name:
                                    self.recent_confidences.append(rec_confidence)
                                    if len(self.recent_confidences) > self.confidence_history_size:
                                        self.recent_confidences.pop(0)

                                    avg_confidence = sum(self.recent_confidences) / len(self.recent_confidences)

                                    if avg_confidence >= 75:
                                        color = (0, 255, 0)  # Green
                                        self.master_present = True
                                        self.last_activity_time = time.time()
                                        self.prevent_sleep()

                                        label = f"{identity}: {rec_confidence:.1f}% (avg: {avg_confidence:.1f}%)"

                                        # Show blacklist message on screen instead of using input()
                                        cv2.putText(display_frame, "Press 'B' to Blacklist", 
                                                    (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                                        # Check if the user pressed 'B' (blacklist)
                                        key = cv2.waitKey(1) & 0xFF
                                        if key == ord('b'):
                                            self.add_to_blacklist(face_resized)
                                            print("⚠️ Face has been added to the blacklist!")

                                    else:
                                        color = (0, 0, 255)  # Red
                                        label = f"Unknown: {rec_confidence:.1f}%"

                                # Draw colored rectangle based on match
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(display_frame, label, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display status with percentage
                if self.master_present and len(self.recent_confidences) > 0:
                    avg_percentage = sum(self.recent_confidences) / len(self.recent_confidences)
                    status = f"Master Present - Confidence: {avg_percentage:.1f}%"
                    color = (0, 255, 0)
                else:
                    status = "Master Absent"
                    color = (0, 0, 255)
                
                cv2.putText(display_frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show frame
                cv2.imshow("Face Recognition", display_frame)
                
                # Check for quit command
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
        finally:
            # Clean up
            self.running = False
            if self.lock_control_thread.is_alive():
                self.lock_control_thread.join(timeout=1)
            
            self.allow_sleep()
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()
            print("Face recognition system stopped")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Face Recognition System")
    parser.add_argument("--model", type=str, default="yolov8n-face.pt",
                       help="Path to YOLOv8 face detection model")
    parser.add_argument("--data-dir", type=str, default="face_data",
                       help="Directory to store face recognition data")
    parser.add_argument("--master", type=str, default="master",
                       help="Name for the master user")
    parser.add_argument("--timeout", type=int, default=5,
                       help="Inactivity timeout in seconds before locking")
    parser.add_argument("--tolerance", type=float, default=0.4,
                       help="Recognition confidence tolerance (lower = more sensitive)")
    parser.add_argument("--retrain", action="store_true",
                       help="Force retraining of the master profile")
    
    args = parser.parse_args()
    
    # Delete existing profile if retrain is requested
    if args.retrain:
        profile_path = os.path.join(args.data_dir, f"{args.master}_knn_model.pkl")
        embeddings_path = os.path.join(args.data_dir, f"{args.master}_embeddings.pkl")
        
        if os.path.exists(profile_path):
            os.remove(profile_path)
            print(f"Removed existing profile: {profile_path}")
            
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
            print(f"Removed existing embeddings: {embeddings_path}")
    
    # Create and run the face recognition system
    system = FaceRecognitionSystem(
        model_path=args.model,
        data_dir=args.data_dir,
        master_name=args.master,
        inactivity_timeout=args.timeout,
        recognition_tolerance=args.tolerance
    )
    
    system.run()

if __name__ == "__main__":
    main()