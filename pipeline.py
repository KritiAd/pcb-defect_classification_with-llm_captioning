from LLM import generate_response_openai
from yolo_pred import test_trained_model
import streamlit

best_model_path = "/home/kriti/projects/week-3/runs/classify/pcb_defects_v1/weights/best.pt"
input_image_path = "/home/kriti/projects/week-3/images_split/val/Mouse_bite/01_mouse_bite_07.jpg"

def get_result(image_path):

    top_class , confindence_score = test_trained_model(best_model_path, image_path)

    prompt = f"Given an image of a PCB defect, predicted the defect is: {top_class},  with a confidence score of {confindence_score:.2f}. Write a concise caption describing this defect in a factual tone."

    response = generate_response_openai(image_path=image_path, prompt=prompt)

    return response.choices[0].message.content


print(get_result(input_image_path))