import tensorflow as tf
import numpy as np
import gradio
import gradio as gr


model = tf.keras.models.load_model("hiragana.h5")

output_labels = [
                'あ', 'い', 'う','え', 'お','か', 
                'が', 'き', 'ぎ', 'く', 'ぐ', 
                'け', 'げ', 'こ', 'ご','さ',
                'ざ', 'し', 'じ', 'す', 'ず',
                'せ', 'ぜ', 'そ','ぞ', 'た', 
                'だ', 'ち', 'ぢ', 'つ','づ', 
                'て', 'で', 'と', 'ど', 'な', 
                'に', 'ぬ', 'ね', 'の','は', 
                'ば', 'ぱ', 'ひ', 'び', 'ぴ',
                'ふ', 'ぶ', 'ぷ', 'へ', 'べ', 
                'ぺ', 'ほ', 'ぼ', 'ぽ','ま',
                'み', 'む', 'め', 'も', 'や', 
                'ゆ', 'よ', 'ら', 'り','る', 
                'れ', 'ろ', 'わ','を','ん'
]

def recognize_hira(image):
    image = image.reshape(1,48,48,1)
    prediction = model.predict(image)
    y_class = int(prediction.argmax(axis=-1))
    return str(output_labels[y_class])

sketchpad = gr.inputs.Sketchpad(shape=(48,48))
label = gr.outputs.Label(num_top_classes=1)

hira = gr.Interface(
    recognize_hira, 
    sketchpad, 
    label,
    title="HIRAGANA Sketch Pad",
    description="Trained on the handwritten Hiragana dataset, ETL8G. Write out a hiragana chracter and the model will make a guess.",
    live=True,
    capture_session=True,
)

if __name__ == "__main__":
    hira.launch()