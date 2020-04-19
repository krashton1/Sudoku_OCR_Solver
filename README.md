# SudokuOCRSolver

Sudoku OCR Solver is a python program, capable of taking an input image, processing it, and displaying the results of the solution on top of the original image


## Installation

Tesseract-OCR will need to be installed as a standalone in order for pytesseract to function
Follow the guide [here](https://tesseract-ocr.github.io/tessdoc/Home.html) to install 
move digits.traineddata from the root folder to '(location_of_install)\Tesseract-OCR\tessdata'

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

```
pip install opencv-contrib-python
pip install basicsudoku
pip install numpy
pip install pytesseract
```


## Usage

From root folder, run

```
main.py 'filename'
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License
[MIT](https://choosealicense.com/licenses/mit/)