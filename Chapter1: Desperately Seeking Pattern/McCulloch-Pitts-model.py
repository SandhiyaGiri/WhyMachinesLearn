# Function for summation:
def g_x(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum

# Function for threshold
def f_x(theta,sum,op):
  match op:
    case "OR":
      if sum >= theta:
        return 1
      else:
          return 0
    case "AND":
      if sum == theta:
        return 1
      else:
        return 0
    case "NOT":
      if sum == 0:
        return 1
      else:
        return 0
    case "XOR":
      if sum % 2 == theta:
        return 1
      else:
        return 0
    case "NAND":
      if sum == theta:
        return 0
      else:
        return 1
    case "NOR":
      if sum >= theta:
        return 0
      else:
        return 1
    case "XNOR":
      if sum % 2 == theta:
        return 0
      else:
        return 1
    case default:
      print("Invalid operation")

# Getting logical operation, threshold 'theta' and no. of inputs from the user.
def get_inputs():
    op = input("Enter logical operation: ")
    theta = int(input("Enter threshold: "))
    n = int(input("Enter no. of inputs: "))

    arr = []
    for i in range(n):
        a = int(input("Enter input: "))
        if a == 0 or a == 1:
            arr.append(a)
        else:
            print("Invalid input")
            i = i+1

    sum = g_x(arr)
    return f_x(theta,sum,op)

if __name__ == "__main__":
  choice = 1
  while (choice):
    print("Choose an operation:")
    print("1. OR")
    print("2. AND")
    print("3. NOT")
    print("4. XOR")
    print("5. NAND")
    print("6. NOR")
    print("7. XNOR")
    print("0. Exit")
    ans = get_inputs()
    print("The result is: ", ans)
    print("*"*30)
    choice = int(input("Do you want to continue?(Enter'1' : Yes, '0': No): "))
    if choice == 0:
      print("*"*20, "Terminated", "*"*20)
