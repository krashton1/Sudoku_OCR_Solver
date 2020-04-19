import cv2 as cv
import numpy as np

import operator
import sys
import math
import os

import basicsudoku
from basicsudoku import solvers

import pytesseract

# Whether to use webcam to capture video or use a still image loaded from file
CAPTURE_VIDEO = False

# Set the location of tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def main(argv):
    cap = cv.VideoCapture(0)

    # Set up loop variable
    processPuzzle = True
    if CAPTURE_VIDEO:
        processPuzzle = False
    foundPuzzle = False

    try:
        os.stat("temp")
    except:
        os.mkdir("temp")

    while True:

        # Read Image or Video Stream
        if CAPTURE_VIDEO:
            _, origImage = cap.read()
        else:
            if len(argv) >= 1:
                origImage = cv.imread(str(argv[0]))
            else:
                origImage = cv.imread('puzzle (4).jpg')

        # Add alpha channel to image
        origImage = cv.cvtColor(origImage.copy(), cv.COLOR_RGB2RGBA)
        origImage[:, :, 3] = 0

        if processPuzzle:
            # Show the original image before processing
            cv.imshow("Sudoku", cv.resize(origImage, (450, 600), interpolation=cv.INTER_AREA))

            # Find puzzle within image
            print("Finding Puzzle...")
            puzzleImage, _, unwarpTransf = findPuzzle(origImage)

            # Run OCR on each box within puzzle
            print("Running OCR on Puzzle...")
            unsolvedSudoku = ocrPuzzle(puzzleImage)

            # Print to console the found puzzle
            # for i in range(9):
            #     print(unsolvedSudoku[i])
            # print()

            # Count the number of clues we found while also getting rid of bad reads
            numClues = 0
            for y in range(9):
                for x in range(9):
                    if unsolvedSudoku[y][x] == '1' or unsolvedSudoku[y][x] == '2' or unsolvedSudoku[y][x] == '3' or unsolvedSudoku[y][x] == '4' or unsolvedSudoku[y][x] == '5' or unsolvedSudoku[y][x] == '6' or unsolvedSudoku[y][x] == '7' or unsolvedSudoku[y][x] == '8' or unsolvedSudoku[y][x] == '9':
                        numClues += 1
                    else:
                        unsolvedSudoku[y][x] = '_'

            # Check if our found puzzle is valid
            if isFoundPuzzleValid(unsolvedSudoku) == False:
                print('Something went wrong in OCR, Puzzle was found to be invalid.\nThis is what OCR found')
                for i in range(9):
                    print(unsolvedSudoku[i])
                return

            # Minimum number of clues to solve a sudoku is 17, any less and our sudoku is invalid
            if numClues >= 17:

                # Solve the Puzzle
                print("Solving Puzzle...")
                solvedSudoku = solveSudoku(unsolvedSudoku)

                # We are assuming that if we have a valid puzzle, a valid soln exists. BAD ASSUMPTION
                foundPuzzle = True
            else:
                print("not enough given digits.\n OCR may have missed digits if resolution is not high, or if puzzle plane is very skewed.\nThis is what OCR found")
                for i in range(9):
                    print(unsolvedSudoku[i])
                return

            # Exit loop
            processPuzzle = False

        # If a puzzle has been previously found and cached, print result to screen
        if foundPuzzle:
            # Print the solved sudoku back onto the original image
            print("Solution Found")
            origImage = printPuzzle(origImage, puzzleImage, unwarpTransf, solvedSudoku, unsolvedSudoku)

        # Display final result
        # cv.imshow("Sudoku", origImage)
        cv.imshow("Sudoku", cv.resize(origImage, (450, 600), interpolation=cv.INTER_AREA))
        cv.imwrite("SolvedPuzzle.jpg", origImage)

        # Exit loop if ESC is pressed, or if processing image and not a puzzle
        k = cv.waitKey(1)
        if k == 27 or CAPTURE_VIDEO == False:
            break
        elif k == 32: # Process a puzzle when space is pressed
            processPuzzle = True

    # Exit program if ESC is pressed, or window is closed
    print("\nPress ESC to quit")
    k1 = cv.waitKey(0)
    k2 = cv.getWindowProperty("Sudoku", cv.WND_PROP_VISIBLE)
    if k1 == 27 or 1 < 1.0:
        cv.destroyAllWindows()


#####################################################################################


# Find sudoku puzzle within image
def findPuzzle(origImage):

    # Process the image so we can find the puzzle within the image
    # The puzzle should be the largest contained shape within the image
    processedImage = cv.cvtColor(origImage.copy(), cv.COLOR_BGR2GRAY)
    proccesedImage = cv.GaussianBlur(processedImage, (5, 5), 0)
    proccesedImage = cv.adaptiveThreshold(proccesedImage, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    proccesedImage = cv.bitwise_not(proccesedImage, proccesedImage)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    proccesedImage = cv.dilate(proccesedImage, kernel)

    #cv.imshow('Processed Image', proccesedImage)

    # Find all the contained shapes within the image
    shapes, _ = cv.findContours(proccesedImage.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    shapes = sorted(shapes, key=cv.contourArea, reverse=True)
    largestShape = shapes[0]

    # Find the corners of the shape
    ptTopLeft, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largestShape]), key=operator.itemgetter(1))
    ptTopRight, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largestShape]), key=operator.itemgetter(1))
    ptBotLeft, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largestShape]), key=operator.itemgetter(1))
    ptBotRight, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largestShape]), key=operator.itemgetter(1))
    croppedRect = [largestShape[ptTopLeft][0], largestShape[ptTopRight][0], largestShape[ptBotRight][0], largestShape[ptBotLeft][0]]
    ptTopLeft, ptTopRight, ptBotRight, ptBotLeft = croppedRect[0], croppedRect[1], croppedRect[2], croppedRect[3]

     # Find the max side length of the warp transform
    maxSideLength = max([
        distance(ptBotRight, ptTopRight),
        distance(ptTopLeft, ptBotLeft),
        distance(ptBotRight, ptBotLeft),
        distance(ptTopLeft, ptTopRight)
    ])

    # Calculate the perspective transforms
    normalTransf = np.array([ptTopLeft, ptTopRight, ptBotRight, ptBotLeft], dtype='float32')
    warpedTransf = np.array( [[0, 0], [maxSideLength - 1, 0], [maxSideLength - 1, maxSideLength - 1], [0, maxSideLength - 1]], dtype='float32')
    perspectiveTransf = cv.getPerspectiveTransform(normalTransf, warpedTransf)

    # Warp Image
    warpedImage = cv.warpPerspective(origImage, perspectiveTransf, (int(maxSideLength), int(maxSideLength)))
    #cv.imshow("warped image", warpedImage)

    # Unwarp Image
    unwarpTransf = cv.getPerspectiveTransform(warpedTransf, normalTransf)
    # unwarpedImage = cv.warpPerspective(warpedImage, unwarpTransf, (origImage.shape[0], origImage.shape[1]))
    #cv.imshow("unwarped image", unwarpedImage)

    return warpedImage, perspectiveTransf, unwarpTransf


#####################################################################################


# Run OCR on each box within puzzle
def ocrPuzzle(puzzle):
    # Split puzzle into 9x9 individual boxes and OCR each box
    boxDimensions = (puzzle.shape[0]/9, puzzle.shape[1]/9)
    boxImages = [[], [], [], [], [], [], [], [], []]
    sudokuGrid = [[], [], [], [], [], [], [], [], []]
    for y in range(9):
        for x in range(9):
            boxImg = puzzle[(int)(y * boxDimensions[1]): (int)((y+1) * boxDimensions[1]), (int)(x * boxDimensions[0]): (int)((x+1) * boxDimensions[0])]
            boxX = boxImg.shape[0]/10
            boxY = boxImg.shape[1]/10
            boxImg = boxImg[(int)((float)(boxY * 2)): (int)((float)(boxY * 8)),(int)((float)(boxX * 2)): (int)((float)(boxX * 8))]
            # cv.imshow("box" + str(x) + str(y), boxImg)

            # Filter image to prep for OCR
            boxImages[y].append(cv.cvtColor(boxImg, cv.COLOR_BGR2GRAY))
            boxImages[y][x] = cv.GaussianBlur(boxImages[y][x], (5, 5), 0)
            boxImages[y][x] = cv.threshold(boxImages[y][x], 120, 255, cv.THRESH_BINARY)[1]
            # boxImages[y][x] = cv.adaptiveThreshold(boxImages[y][x], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 12)
            cv.imwrite("temp/box" + str(y) + "" + str(x) + ".png", boxImages[y][x])

            # OCR on box image
            digit = pytesseract.image_to_string("temp/box" + str(y) + "" + str(x) + ".png", lang='digits', config='--psm 10  --oem 3 -c tessedit_char_whitelist=123456789')
            if digit == '':
                digit = '_'
            sudokuGrid[y].append(digit)
            # cv.imshow("box",boxImg)

    return sudokuGrid


#####################################################################################


# Print the solved sudoku back onto the original puzzle image
def printPuzzle(origImage, puzzle, unwarpTransf, solvedSudoku, unsolvedSudoku):
    # Print solution back onto puzzle
    for yy in range(9):
        for xx in range(9):
            if unsolvedSudoku[yy][xx] == '_':
                puzzle = cv.putText(puzzle, str(solvedSudoku[yy][xx]), ((int)(xx * puzzle.shape[0] / 9 + puzzle.shape[0] / 9 / 3), (int)(yy * puzzle.shape[1] / 9 + puzzle.shape[1] / 9 - puzzle.shape[1] / 9 / 6)), cv.FONT_HERSHEY_TRIPLEX, (int)(puzzle.shape[1] / 9 / 40), (0, 0, 255), 2, cv.LINE_AA)

    # Overlay the modified warped image back onto the original image
    unwarpedImage = cv.warpPerspective(puzzle, unwarpTransf, (origImage.shape[1], origImage.shape[0]), borderValue=(255, 255, 255, 255))
    alpha = 1.0 - unwarpedImage[:, :, 3] / 255.0
    for t in range(0, 3):
        origImage[0:unwarpedImage.shape[0], 0:unwarpedImage.shape[1], t] = (alpha * unwarpedImage[:, :, t] + (1 - alpha) * origImage[0:unwarpedImage.shape[0], 0:unwarpedImage.shape[1], t])

    return origImage


#####################################################################################


def solveSudoku(unsolvedSudoku):
    solvedSudoku = [[], [], [], [], [], [], [], [], []]
    board = basicsudoku.SudokuBoard()
    # board.strict = False
    for x in range(9):
        for y in range(9):
            if unsolvedSudoku[y][x] != '_':
                board[x, y] = (int)(unsolvedSudoku[y][x])

    print('Found Puzzle')
    print(board)
    print()

    solvers.BasicSolver(board)

    print('Solved Puzzle')
    print(board)
    print()

    for y in range(9):
        for x in range(9):
            solvedSudoku[y].append(board[x, y])

    # Print to console the found puzzle
    # for i in range(9):
    #     print(solvedSudoku[i])

    return solvedSudoku


#####################################################################################


def isFoundPuzzleValid(unsolvedSudoku):
    # Check if all columns are valid
    for x in range(9):
        for t in range(1, 10):
            numT = 0
            for y in range(9):
                if unsolvedSudoku[y][x] == str(t):
                    numT += 1
            if numT > 1:
                return False

    # Check if all rows are valid
    for y in range(9):
        for t in range(1, 10):
            numT = 0
            for x in range(9):
                if unsolvedSudoku[y][x] == str(t):
                    numT += 1
            if numT > 1:
                return False

    # Check if all boxes are valid
    for y in range(3):
        for x in range(3):
            for t in range(1, 10):
                numT = 0
                for yy in range(3):
                    for xx in range(3):
                        if unsolvedSudoku[yy + 3 * y][xx + 3 * x] == str(t):
                            numT += 1
                if numT > 1:
                    return False

    return True


#####################################################################################


# Find distance between 2 points
def distance(pt1, pt2):
    a = pt1[0] - pt2[0]
    b = pt1[1] - pt2[1]
    return math.sqrt(a ** 2 + b ** 2)


#####################################################################################


if __name__ == "__main__":
    main(sys.argv[1:])
