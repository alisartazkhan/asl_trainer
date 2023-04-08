# ASL Hand Gesture Recognition

This program uses computer vision techniques to track the left hand of an individual using the front-facing camera and correctly identify any number from 1-10 represented using American Sign Language (ASL).

The program first detects all the 22 points in the person's finger using the Mediapipe library. Then, it checks the orientation of the fingers and displays the appropriate number image represented by the person.

## Usage

To use the program, simply run the `main()` function. The program will use the front-facing camera to detect hand gestures and display the corresponding number image on the screen.

## Required Libraries

The program requires the following libraries to be installed:

- opencv-python
- mediapipe

These libraries can be installed using pip.

## Files

- `HandTracker.py`: The main program file containing all the code.
- `FingerImages`: A directory containing all the pictures of ASL gestures 1-10 used by the program.

## Future Work

This program can be further improved by using deep learning models such as convolutional neural networks (CNNs) to improve the accuracy of the recognition. Additionally, more gestures can be added to the dataset to recognize a wider range of hand gestures.
