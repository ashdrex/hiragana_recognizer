import tensorflow as tf
import gradio
import gradio as gr


model = tf.keras.models.load_model("hiragana.h5")


output_labels = {
0: 'あ', 1: 'い', 2: 'う', 3:'え', 4: 'お',5: 'か', 
6: 'が', 7: 'き', 8:'ぎ', 9: 'く', 10:'ぐ', 
11: 'け', 12: 'げ', 13: 'こ', 14:'ご', 15:'さ',
16:'ざ', 17: 'し', 18: 'じ', 19: 'す', 20: 'ず',
21: 'せ', 22:'ぜ', 23:'そ',24: 'ぞ', 25: 'た', 
26: 'だ', 27:'ち', 28: 'ぢ', 29: 'つ', 30:'づ', 
31:'て', 32: 'で', 33: 'と', 34: 'ど', 35:'な', 
36: 'に', 37:'ぬ', 38: 'ね', 39:'の', 40: 'は', 
41: 'ば', 42:'ぱ', 43:'ひ', 44: 'び', 45: 'ぴ',
46: 'ふ', 47:'ぶ', 48:'ぷ', 49:'へ', 50: 'べ', 
51:'ぺ', 52: 'ほ', 53:'ぼ', 54:'ぽ', 55:'ま', 
56: 'み', 57:'む', 58: 'め', 59:'も', 60: 'や', 
61: 'ゆ', 62: 'よ', 63: 'ら', 64: 'り',65: 'る', 
66:'れ', 67: 'ろ', 68: 'わ', 69:'を', 70:'ん'}

def recognize_hira(image):
    # prediction = model.predict(image.reshape(1,48,48,1)).tolist()[0]
    # for i in range(71):
    #     print(i, prediction[i])
    # return {output_labels[i]: prediction[i] for i in range(71)}
        # prediction = model.predict(image.reshape(1,48,48,1)).tolist()[0]
    image = image.reshape(1,48,48,1)
    prediction = model.predict(image)
    y_class = int(prediction.argmax(axis=-1))
    print(y_class)
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