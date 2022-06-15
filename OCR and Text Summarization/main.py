import argparse
from PIL import Image

from text_summarization import TextSummarization
from OCR import OCR

from PathType import PathType

def main(opt):
    text_summarization = TextSummarization()
    ocr = OCR()

    img_file = Image.open(opt.input).convert('RGB')

    text_on_img = ocr.image_to_string(img_file)

    original_sentences, best_sentences, _ = text_summarization.summarize(text_on_img, opt.sentences)

    text_summarization.save_summary(original_sentences, best_sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", 
                        required=True,
                        help="Path of input", 
                        type = PathType(exists=True, type='file'))

    parser.add_argument('--sentences', 
                        help='Max value of sentences to include on summary (default: 6)', 
                        type=int, 
                        default=6)

    parser.add_argument('--save', action ='store_true', help='Save result')

    opt = parser.parse_args()
    
    main(opt) 