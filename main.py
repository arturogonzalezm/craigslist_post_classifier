import sys

from src.craigslist_post_classifier import main

if __name__ == "__main__":
    training_file = 'data/training.json'
    main(training_file, sys.stdin)
