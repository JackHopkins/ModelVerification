# InterpBench tasks

All 86 models from [cybershiptrooper/InterpBench](https://huggingface.co/cybershiptrooper/InterpBench),
mirrored under `tasks/<case_id>/`. Each task directory contains:

- `ll_model.pth` — trained transformer weights (PyTorch / TransformerLens `HookedTransformer` state dict)
- `ll_model_cfg.pkl` — pickled `HookedTransformerConfig`
- `edges.pkl` — the ground-truth circuit (set of edges in the computational graph implementing the task)
- `meta.json` — SIIT training hyperparameters

| Case | Task | Layers | d_model | Heads | d_mlp | Params |
|---|---|---|---|---|---|---|
| 2 | Reverse the input sequence. | 4 | 56 | 4 | 224 | 150528 |
| 3 | Returns the fraction of 'x' in the input up to the i-th position for all i. | 2 | 12 | 4 | 48 | 3456 |
| 4 | Return fraction of previous open tokens minus the fraction of close tokens. | 2 | 20 | 4 | 80 | 9600 |
| 7 | Returns the number of times each token occurs in the input. | 2 | 17 | 4 | 68 | 6800 |
| 8 | Identity | 2 | 4 | 4 | 16 | 384 |
| 11 | Counts the number of words in a sequence based on their length. | 2 | 12 | 4 | 48 | 3456 |
| 13 | Analyzes the trend (increasing, decreasing, constant) of numeric tokens. | 2 | 20 | 4 | 80 | 9600 |
| 14 | Returns the count of 'a' in the input sequence. | 2 | 8 | 4 | 32 | 1536 |
| 15 | Returns each token multiplied by two and subtracted by its index. | 3 | 4 | 4 | 16 | 576 |
| 18 | Classify each token based on its frequency as 'rare', 'common', or 'frequent'. | 2 | 26 | 4 | 104 | 15808 |
| 19 | Removes consecutive duplicate tokens from a sequence. | 2 | 32 | 4 | 128 | 24576 |
| 20 | Detect spam messages based on appearance of spam keywords. | 2 | 13 | 4 | 52 | 3952 |
| 21 | Extract unique tokens from a string | 4 | 50 | 4 | 200 | 118400 |
| 24 | Identifies the first occurrence of each token in a sequence. | 2 | 36 | 4 | 144 | 31104 |
| 25 | Normalizes token frequencies in a sequence to a range between 0 and 1. | 2 | 62 | 4 | 248 | 91264 |
| 26 | Creates a cascading effect by repeating each token in sequence incrementally. | 2 | 21 | 4 | 84 | 10416 |
| 29 | Creates abbreviations for each token in the sequence. | 2 | 13 | 4 | 52 | 3952 |
| 30 | Tags numeric tokens in a sequence based on whether they fall within a given range. | 2 | 4 | 4 | 16 | 384 |
| 31 | Identify if tokens in the sequence are anagrams of the word 'listen'. | 2 | 4 | 4 | 16 | 384 |
| 33 | Checks if each token's length is odd or even. | 2 | 4 | 4 | 16 | 384 |
| 34 | Calculate the ratio of vowels to consonants in each word. | 2 | 16 | 4 | 64 | 6144 |
| 35 | Alternates capitalization of each character in words. | 2 | 9 | 4 | 36 | 1872 |
| 36 | Classifies each token as 'positive', 'negative', or 'neutral' based on emojis. | 2 | 6 | 4 | 24 | 768 |
| 37 | Reverses each word in the sequence except for specified exclusions. | 2 | 12 | 4 | 48 | 3456 |
| 39 | Returns the fraction of 'x' in the input up to the i-th position for all i. | 2 | 120 | 4 | 480 | 345600 |
| 40 | Sum the last and previous to last digits of a number | 2 | 4 | 4 | 16 | 384 |
| 41 | Make each element of the input sequence absolute | 2 | 4 | 4 | 16 | 384 |
| 43 | Returns the corresponding Fibonacci number for each element in the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 44 | Replaces each element with the number of elements greater than it in the sequence | 2 | 24 | 4 | 96 | 13824 |
| 45 | Doubles the first half of the sequence | 3 | 24 | 4 | 96 | 20736 |
| 46 | Decrements each element in the sequence by 1 | 2 | 4 | 4 | 16 | 384 |
| 49 | Decrements each element in the sequence until it becomes a multiple of 3. | 2 | 4 | 4 | 16 | 384 |
| 50 | Applies the hyperbolic cosine to each element | 2 | 4 | 4 | 16 | 384 |
| 51 | Checks if each element is a Fibonacci number | 2 | 4 | 4 | 16 | 384 |
| 52 | Takes the square root of each element. | 2 | 4 | 4 | 16 | 384 |
| 53 | Increment elements at odd indices by 1 | 2 | 4 | 4 | 16 | 384 |
| 54 | Applies the hyperbolic tangent to each element. | 2 | 4 | 4 | 16 | 384 |
| 55 | Applies the hyperbolic sine to each element. | 2 | 4 | 4 | 16 | 384 |
| 56 | Sets every third element to zero. | 2 | 4 | 4 | 16 | 384 |
| 58 | Mirrors the first half of the sequence to the second half. | 3 | 32 | 4 | 128 | 36864 |
| 60 | Increment each element in the sequence by 1. | 2 | 4 | 4 | 16 | 384 |
| 62 | Replaces each element with its factorial. | 2 | 4 | 4 | 16 | 384 |
| 63 | Replaces each element with the number of elements less than it in the sequence. | 2 | 24 | 4 | 96 | 13824 |
| 64 | Cubes each element in the sequence. | 2 | 4 | 4 | 16 | 384 |
| 65 | Calculate the cube root of each element in the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 66 | Round each element in the input sequence to the nearest integer. | 2 | 4 | 4 | 16 | 384 |
| 67 | Multiply each element of the sequence by the length of the sequence. | 2 | 24 | 4 | 96 | 13824 |
| 68 | Increment each element until it becomes a multiple of 3 | 2 | 4 | 4 | 16 | 384 |
| 69 | Assign -1, 0, or 1 to each element of the input sequence based on its sign. | 2 | 4 | 4 | 16 | 384 |
| 70 | Apply the cosine function to each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 71 | Divide each element by the length of the sequence | 2 | 24 | 4 | 96 | 13824 |
| 72 | Negate each element in the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 73 | Apply the sine function to each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 75 | Double each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 77 | Apply the tangent function to each element of the sequence. | 2 | 4 | 4 | 16 | 384 |
| 79 | Check if each number in a sequence is prime | 2 | 4 | 4 | 16 | 384 |
| 80 | Subtract a constant from each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 82 | Halve the elements in the second half of the sequence. | 4 | 24 | 4 | 96 | 27648 |
| 83 | Triple each element in the sequence. | 2 | 4 | 4 | 16 | 384 |
| 84 | Apply the arctangent function to each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 85 | Square each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 86 | Check if each element is a power of 2. Return 1 if true, otherwise 0. | 2 | 4 | 4 | 16 | 384 |
| 87 | Binarize a sequence of integers using a threshold. | 2 | 4 | 4 | 16 | 384 |
| 90 | Replaces a specific token with another one. | 2 | 4 | 4 | 16 | 384 |
| 91 | Set all values below a threshold to 0 | 2 | 4 | 4 | 16 | 384 |
| 93 | Swaps the nth with the n+1th element if n%2==1. | 3 | 20 | 4 | 80 | 14400 |
| 95 | Counts the distinct prime factors of each number in the input list. | 2 | 4 | 4 | 16 | 384 |
| 97 | Scale a sequence by its maximum element. | 3 | 200 | 4 | 800 | 1440000 |
| 101 | Check if each element is a square of an integer. | 2 | 4 | 4 | 16 | 384 |
| 102 | Reflects each element within a range (default is [2, 7]). | 2 | 4 | 4 | 16 | 384 |
| 103 | Swap consecutive numbers in a list | 3 | 24 | 4 | 96 | 20736 |
| 104 | Apply exponential function to all elements of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 105 | Replaces each number with the next prime after that number. | 2 | 4 | 4 | 16 | 384 |
| 106 | Sets all elements to zero except for the element at index 1. | 2 | 4 | 4 | 16 | 384 |
| 110 | Inserts zeros between each element, removing the latter half of the list. | 2 | 20 | 4 | 80 | 9600 |
| 111 | Returns the last element of the sequence and pads the rest with zeros. | 3 | 24 | 4 | 96 | 20736 |
| 113 | Inverts the sequence if it is sorted in ascending order, otherwise leaves it unchanged. | 7 | 88 | 4 | 352 | 650496 |
| 114 | Apply a logarithm base 10 to each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 121 | Compute arcsine of all elements in the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 122 | Check if each number is divisible by 3. | 2 | 4 | 4 | 16 | 384 |
| 123 | Apply arccosine to each element of the input sequence. | 2 | 4 | 4 | 16 | 384 |
| 124 | Check if all elements in a list are equal. | 3 | 24 | 4 | 96 | 20736 |
| 129 | Checks if all elements are a multiple of n (set the default at 2). | 3 | 4 | 4 | 16 | 576 |
| 130 | Clips each element to be within a range (make the default range [2, 7]). | 3 | 4 | 4 | 16 | 576 |
| ioi | Indirect Object Identification (IOI) task. | 6 | 64 | 4 | 3072 | 84934656 |
| ioi_next_token | Indirect Object Identification (IOI) task, trained using next token prediction. | 6 | 64 | 4 | 3072 | 2457600 |
