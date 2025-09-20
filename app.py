from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image

# ======== Import your own model and utilities ========
from nlp_utils import clean_sentence
from vocabulary import Vocabulary
# from model import EncoderCNN, DecoderRNN  # Uncomment and adjust based on your model filenames

# ======== Flask config ========
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# ======== Load Vocabulary ========
vocab = Vocabulary(
    vocab_threshold=5,
    vocab_file="./vocab.pkl",
    annotations_file="C:/Users/santo/Downloads/MS COCO Dataset/captions/captions_train2014.json",
    vocab_from_file=True
)

# ======== Load your trained model ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example — replace with your actual model loading
# encoder = EncoderCNN(embed_size).eval().to(device)
# decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).eval().to(device)
# encoder.load_state_dict(torch.load("encoder.ckpt", map_location=device))
# decoder.load_state_dict(torch.load("decoder.ckpt", map_location=device))

def allowed_file(filename):
    return '.' in filename and \           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ======== Home page ========
@app.route('/')
def index():
    return render_template('index.html')

# ======== Handle upload ========
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ===== Generate caption =====
        caption = generate_caption(filepath)

        return render_template('result.html', filename=filename, caption=caption)
    return redirect(url_for('index'))

# ======== Generate caption function ========
def generate_caption(image_path):
    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Here you would use your encoder + decoder to get output
    # Example dummy output
    output = [vocab('<start>'), vocab('a'), vocab('man'), vocab('riding'), vocab('bike'), vocab('<end>')]
    sentence = clean_sentence(output, vocab.idx2word)
    
    return sentence

# ======== Serve uploaded images ========
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# ======== Run app ========
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
