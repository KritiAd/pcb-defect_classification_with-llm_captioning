from ultralytics import YOLO
import os


def train_yolo_simple(dataset_path):
  """
  Train YOLOv11 classifier with automatic train/val split
  """
 
  model = YOLO('yolo11n-cls.pt')  

  results = model.train(
      data=dataset_path,          
      epochs=50,                 
      imgsz=320,                 
      batch=8,                   
      split=0.8,                 
      device='cpu',              
      workers=2,                 
      patience=10,               
      save=True,                 
      plots=True,                
      name='pcb_defects_v1'     
  )
  print(f"Training completed!")
  print(f"Best model saved at: {model.trainer.save_dir}")
  return model, results

def test_trained_model(model_path, test_image_path):
  """
  Test your trained model on a single image
  """
  
  model = YOLO(model_path)
  
  results = model(test_image_path)
  result = results[0]
  
  names = result.names
  probs = result.probs
  top_class = names[probs.top1]
  confidence = probs.top1conf.item()
  print(f"\nPrediction Results:")
  print(f"Image: {test_image_path}")
  print(f"Predicted Class: {top_class}")
  print(f"Confidence: {confidence:.4f}")

  print(f"\nAll class probabilities:")
  for i, prob in enumerate(probs.data):
      print(f"  {names[i]}: {prob:.4f}")

  return top_class, confidence

# USAGE EXAMPLE:
if __name__ == "__main__":
   
   model, results = train_yolo_simple("images")
  
