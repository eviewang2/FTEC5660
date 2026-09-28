# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.
### Chain design

```mermaid
flowchart TD
    A[Receipt images] --> B[ChatPromptTemplate<br/>system rules + receipt image]
    B -->|chain.batch, all receipts in parallel| C[deepseek-v4-flash-vision-exp]
    C --> D[JsonOutputParser<br/>items, discounts, subtotal, rounding, payment_amount]
    D --> E{Self-check in Python<br/>sum items - sum discounts = subtotal?}
    E -->|No: re-read with feedback, up to 2 times| C
    E -->|Yes| F[Sum with Decimal]
    F --> G[Q1: sum of payment_amount<br/>Q2: sum of subtotal + discounts]
```


### Description

My chain has three parts: a prompt template, the DeepSeek model and a JSON parser
(`ChatPromptTemplate | ChatDeepSeek | JsonOutputParser`).

For each receipt, I send the image to the model. The prompt asks the model to read
the receipt and return JSON with five things: all item prices, all discounts, the
subtotal, the rounding and the final payment. I use `chain.batch` so that all receipts
are read at the same time.

The model only reads the numbers. Python does all the maths with `Decimal`, so there
are no rounding errors.

When I tested my code, I found a problem. Sometimes the model read a number wrongly,
for example 6.00 as 5.00, because the photo was not clear. So I added a self-check.
For each receipt, Python checks if "items minus discounts equals subtotal". If not,
the model reads the receipt again, up to two more times.

- Question 1 = the sum of the final payments.
- Question 2 = the sum of (subtotal + discounts) for each receipt.

After adding the self-check, my code gave the correct answers in all three test runs.