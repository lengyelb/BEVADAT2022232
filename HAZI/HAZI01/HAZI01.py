# Create a function that returns with a subset of a list.
# The subset's starting and ending indexes should be set as input parameters (the list as well).
# return type: list
# function name must be: subset
# input parameters: input_list,start_index,end_index
def subset(input_list, start_index, end_index) -> list:
    return input_list[start_index:end_index]


# Create a function that returns every nth element of a list.
# return type: list
# function name must be: every_nth
# input parameters: input_list,step_size
def every_nth(input_list, step_size) -> list:
    return [input_list[i] for i in range(0, len(input_list), step_size)]


# Create a function that can decide whether a list contains unique values or not
# return type: bool
# function name must be: unique
# input parameters: input_list
def unique(input_list) -> bool:
    return len(set(input_list)) == len(input_list)


# Create a function that can flatten a nested list ([[..],[..],..])
# return type: list
# function name must be: flatten
# input parameters: input_list
def flatten(input_list) -> list:
    return [item for sublist in input_list for item in sublist]


# Create a function that concatenates n lists
# return type: list
# function name must be: merge_lists
# input parameters: *args
def merge_lists(*args) -> list:
    return [item for sublist in args for item in sublist]


# Create a function that can reverse a list of tuples
# example [(1,2),...] => [(2,1),...]
# return type: list
# function name must be: reverse_tuples
# input parameters: input_list
def reverse_tuples(input_list) -> list[tuple]:
    return [(item[1], item[0]) for item in input_list]


# Create a function that removes duplicates from a list
# return type: list
# function name must be: remove_duplicates
# input parameters: input_list
def remove_duplicates(input_list) -> list:
    return list(set(input_list))


# Create a function that transposes a nested list (matrix)
# return type: list
# function name must be: transpose
# input parameters: input_list
def transpose(input_list) -> list[list]:
    return [[input_list[j][i] for j in range(len(input_list))] for i in range(len(input_list[0]))]


# Create a function that can split a nested list into chunks
# chunk size is given by parameter
# return type: list
# function name must be: split_into_chunks
# input parameters: input_list,chunk_size
def split_into_chunks(input_list, chunk_size) -> list[list]:
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


# Create a function that can merge n dictionaries
# return type: dictionary
# function name must be: merge_dicts
# input parameters: *dict
def merge_dicts(*dict_args) -> dict:
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# Create a function that receives a list of integers and sort them by parity
# and returns with a dictionary like this: {"even":[...],"odd":[...]}
# return type: dict
# function name must be: by_parity
# input parameters: input_list
def by_parity(input_list) -> dict:
    result = {}
    for item in input_list:
        if item % 2 == 0:
            result["even"].append(item)
        else:
            result["odd"].append(item)
    return result


# Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
# and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
# in short calculates the mean of the values key wise
# return type: dict
# function name must be: mean_key_value
# input parameters: input_dict
def mean_key_value(input_dict) -> dict:
    result = {}
    for key, value in input_dict.items():
        result[key] = sum(value) / len(value)
    return result

# If all the functions are created convert this notebook into a .py file and push to your repo
