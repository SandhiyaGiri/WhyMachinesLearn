import streamlit as st

# Function for summation
def g_x(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Function for threshold
def f_x(theta, total, op):
    if op == "OR":
        return 1 if total >= theta else 0
    elif op == "AND":
        return 1 if total == theta else 0
    elif op == "NOT":
        return 1 if total == 0 else 0
    elif op == "XOR":
        return 1 if total % 2 == theta else 0
    elif op == "NAND":
        return 0 if total == theta else 1
    elif op == "NOR":
        return 0 if total >= theta else 1
    elif op == "XNOR":
        return 0 if total % 2 == theta else 1
    else:
        return "Invalid operation"

# Streamlit app setup
st.title("Logical Operations using McCulloch-Pitts model")

# Dropdown for selecting operation
operation = st.selectbox(
    "Select a logical operation:",
    ["OR", "AND", "NOT", "XOR", "NAND", "NOR", "XNOR"]
)

# Input for threshold value
theta = st.number_input("Enter the threshold value (theta):", min_value=0, value=1, step=1)

# Input for number of inputs (n)
n = st.number_input("Enter the number of inputs (n):", min_value=1, value=1, step=1)

# Input values as a list
st.write("Enter the input values (0 or 1):")
input_values = []
for i in range(n):
    value = st.number_input(f"Input {i + 1}:", min_value=0, max_value=1, value=0, step=1)
    input_values.append(value)

# Calculate the sum and result when the button is clicked
if st.button("Calculate"):
    # Calculate the sum of input values
    total = g_x(input_values)

    # Perform the logical operation
    result = f_x(theta, total, operation)

    # Display results
    st.write(f"Sum of inputs: {total}")
    st.write(f"Result of the operation ({operation}): {result}")
