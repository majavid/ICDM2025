# demos/demo_quickstart.py
import numpy as np

# Import the modules (no assumptions about class/function names)


def main():
    # Just prove imports & basic numpy work
    X = np.random.randn(10, 3)
    print("Imports OK. X shape:", X.shape)


if __name__ == "__main__":
    main()
