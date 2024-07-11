#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
From data (input-output pairings),
and a set of operators and operands as the only starting point,
write a program that will evolve programmatic solutions,
which take in inputs and generate outputs.

Each program will have 1 numeric input and 1 numeric output.
This is much like regression in our simple case,
though can be generalized much further,
toward arbitrarily large and complex programs.

This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
'''

# %%
import random
from typing import TypedDict
from typing import Optional
import math

# import json

# import math
# import datetime
# import subprocess


# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
OPERATORS = "+-/*"
NUMS = list(range(-100, 101, 1))
POP_SIZE = 200


class Node:
    """
    Example prefix formula:
    Y = ( * ( + 20 45 ) ( - 56 X ) )
    This is it's tree:
       *
      /  \
    +     -
    / \   / \
    20 45 56  X

    root = Node(
        data="*",
        left=Node(data="+", left=Node("20"), right=Node("45")),
        right=Node(data="-", left=Node("56"), right=Node("X")),
    )
    """

    def __init__(
        self, data: str, left: Optional["Node"] = None, right: Optional["Node"] = None
    ) -> None:
        self.data = data
        self.left = left
        self.right = right


class Individual(TypedDict):
    """Type of each individual to evolve"""

    genome: Node
    fitness: float


Population = list[Individual]


class IOpair(TypedDict):
    """Data type for training and testing data"""

    input1: int
    output1: float


IOdata = list[IOpair]


def print_tree(root: Node, indent: str = "") -> None:
    """
    Pretty-prints the data structure in actual tree form.
    >>> print_tree(root=root, indent="")
    """
    if root.right is not None and root.left is not None:
        print_tree(root=root.right, indent=indent + "    ")
        print(indent, root.data)
        print_tree(root=root.left, indent=indent + "    ")
    else:
        print(indent + root.data)


def parse_expression(source_code: str) -> Node:
    """
    Turns prefix code into a tree data structure.
    >>> clojure_code = "( * ( + 20 45 ) ( - 56 X ) )"
    >>> root = parse_expression(clojure_code)
    """
    source_code = source_code.replace("(", "")
    source_code = source_code.replace(")", "")
    code_arr = source_code.split()
    # code_arr_copy = source_code.split()
    return _parse_experession(code_arr)


def _parse_experession(code: list[str]) -> Node:
    """
    The back-end helper of parse_expression.
    Not intended for calling directly.
    Assumes code is prefix notation lisp with space delimeters.
    """
    if code[0] in OPERATORS:
        return Node(
            data=code.pop(0),
            left=_parse_experession(code),
            right=_parse_experession(code),
        )
    else:
        return Node(code.pop(0))


def parse_tree_print(root: Node) -> None:
    """
    Stringifies to std-out (print) the tree data structure.
    >>> parse_tree_print(root)
    """
    if root.right is not None and root.left is not None:
        print(f"( {root.data} ", end="")
        parse_tree_print(root.left)
        parse_tree_print(root.right)
        print(") ", end="")
    else:
        # for the case of literal programs... e.g., `4`
        print(f"{root.data} ", end="")


def parse_tree_return(root: Node) -> str:
    """
    Stringifies to the tree data structure, returns string.
    >>> stringified = parse_tree_return(root)
    """
    if root.right is not None and root.left is not None:
        return f"( {root.data} {parse_tree_return(root.left)} {parse_tree_return(root.right)} )"
    else:
        # for the case of literal programs... e.g., `4`
        return root.data


def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as Node, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[Node, int]
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    >>> ind1 = initialize_individual("( + ( * C ( / 9 5 ) ) 32 )", 0)
    """
    return {"genome": parse_expression(genome), "fitness": fitness}


def initialize_data(input1: int, output1: float) -> IOpair:
    """
    For mypy...
    """
    return {"input1": input1, "output1": output1}


def prefix_to_infix(prefix: str) -> str:
    """
    My minimal lisp on python interpreter, lol...
    >>> C = 0
    >>> print(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )"))
    >>> print(eval(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )")))
    """
    prefix = prefix.replace("(", "")
    prefix = prefix.replace(")", "")
    prefix_arr = prefix.split()
    stack = []
    i = len(prefix_arr) - 1
    while i >= 0:
        if prefix_arr[i] not in OPERATORS:
            stack.append(prefix_arr[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix_arr[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


def get_nodes(root: Node) -> list[Node]:
    node_list: list[Node] = []
    if root.left is not None and root.right is not None:
        node_list.extend(get_nodes(root.left))
        node_list.extend(get_nodes(root.right))

    node_list.append(root)
    return node_list


def put_an_x_in_it_node(root: Node) -> None:
    node_list: list[Node] = get_nodes(root)
    rand_node: Node = random.choice(node_list)
    while rand_node.data in OPERATORS:
        rand_node = random.choice(node_list)

    rand_node.data = "x"
    rand_node.left = None
    rand_node.right = None

    return rand_node


def put_an_x_in_it(formula: str) -> str:
    formula_arr = formula.split()
    while True:
        i = random.randint(0, len(formula_arr) - 1)
        if formula_arr[i] not in OPERATORS:
            formula_arr[i] = "x"
            break
    return " ".join(formula_arr)


def gen_rand_prefix_code(depth_limit: int, rec_depth: int = 0) -> str:
    """
    Generates one small formula,
    from OPERATORS and ints from -100 to 200
    """
    rec_depth += 1
    if rec_depth < depth_limit:
        if random.random() < 0.9:
        # if random.random() <= 1.00:  # this always happens yes
            return (
                random.choice(OPERATORS)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
            )
        else:
            return str(random.randint(-100, 100))
    else:
        return str(random.randint(-100, 100))
 # <<<< DO NOT MODIFY


def initialize_pop(pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
         random.choice-1, string.ascii_letters-1, initialize_individual-n
    Example doctest:
    """
    pop: Population = []
    for _ in range(pop_size):
        ind_str: str = gen_rand_prefix_code(3)
        ind: Individual = initialize_individual(ind_str, 0)
        put_an_x_in_it_node(ind["genome"])
        pop.append(ind)

    return pop


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Example doctest:
    """
    p1_nodes: list[Node] = get_nodes(parent1["genome"])
    p2_nodes: list[Node] = get_nodes(parent2["genome"])
    
    select1_node: Node = random.choice(p1_nodes)
    while select1_node.data not in OPERATORS:
        select1_node = random.choice(p1_nodes)
    select2_node: Node = random.choice(p2_nodes)
    while select2_node.data not in OPERATORS:
        select2_node = random.choice(p2_nodes)

    p1_str: str = parse_tree_return(parent1["genome"])
    p2_str: str = parse_tree_return(parent2["genome"])
    if p1_str == "( + ( - -32 -64 ) ( + 13 x ) )":
        pass
    select1_str: str = " " + parse_tree_return(select1_node) + " "
    select2_str: str = " " + parse_tree_return(select2_node) + " "

    c1_str: str = p1_str.replace(select1_str, select2_str)
    c2_str: str = p2_str.replace(select2_str, select1_str)

    # try this for the fun of it and pray
    # replace every x with a constant
    c1_str = c1_str.replace("x", str(random.choice(NUMS)))
    c2_str = c2_str.replace("x", str(random.choice(NUMS)))

    # fix the edge case where -(...) happens :/
    c1_str = c1_str.replace("-(", "(")
    c2_str = c2_str.replace("-(", "(")
    
    if c1_str == "( + ( + 13 x ) ( + 13 x ) )":
        pass
    if c2_str == "( + ( + 13 x ) ( + 13 x ) )":
        pass
    
    c1: Individual = initialize_individual(c1_str, 0.0)
    c2: Individual = initialize_individual(c2_str, 0.0)

    if 'x' not in c1_str:
        put_an_x_in_it_node(c1["genome"])
    if 'x' not in c2_str:
        put_an_x_in_it_node(c2["genome"])
    # parse_tree_print(c1["genome"])
    # print()
    # parse_tree_print(c2["genome"])
    # print()

    return [c1, c2]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          Basic python, random.random~n/2, recombine pair-n
    """
    children: Population = []
    for p0, p1 in zip(parents[::2], parents[1::2]):
        if random.random() < recombine_rate:
            c0, c1 = recombine_pair(p0, p1)
        else:
            c0, c1 = p0, p1
        children.extend([c0, c1])

    return children


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
    Example doctest:
    """
    if random.random() <= mutate_rate:         
        new_prefix: str = gen_rand_prefix_code(2)
        temp_parent: Individual = initialize_individual(new_prefix, 0.0)
        child, _ = recombine_pair(parent, temp_parent)
        parent = child
            
    return parent


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Example doctest:
    """
    mutants: Population = []
    for child in children:
        mutants.append(mutate_individual(child, mutate_rate))

    return mutants


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual, data formatted as IOdata
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Notes:          train/test format is like PSB2 (see IOdata above)
    Example doctest:
    >>> evaluate_individual(ind1, io_data)
    """
    fitness = 0
    errors = []
    for sub_eval in io_data:
        eval_string = parse_tree_return(individual["genome"]).replace(
            "x", str(sub_eval["input1"])
        )

        # In clojure, this is really slow with subprocess
        # eval_string = "( float " + eval_string + ")"
        # returnobject = subprocess.run(
        #     ["clojure", "-e", eval_string], capture_output=True
        # )
        # result = float(returnobject.stdout.decode().strip())

        # In python, this is MUCH MUCH faster:
        try:
            y = eval(prefix_to_infix(eval_string))
        except ZeroDivisionError:
            y = math.inf

        errors.append(abs(sub_eval["output1"] - y))
    # Higher errors is bad, and longer strings is bad
    fitness = sum(errors) + len(eval_string.split())
    # Higher fitness is worse
    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Example doctest:
    """
    for ind in individuals:
        if ind["fitness"] == 0:
            evaluate_individual(ind, io_data)

    # remove the math.infs
    n: int = 0
    while n < len(individuals):
        if individuals[n]["fitness"] in (math.inf, -math.inf):
            individuals.pop(n)
            n -= 1
        n += 1

    return


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    individuals.sort(key=lambda ind: ind["fitness"]) 

    return


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          Basic python, random.choices-1
    Example doctest:
    """
    parents: Population = []
    fitnessen: list[float] = [ind["fitness"] for ind in individuals]
    fitness_max: int = max(fitnessen)
    adj_fitnessen: list[float] = list(map(lambda num: (fitness_max - num + .00000001), fitnessen))
    parents = random.choices(individuals, adj_fitnessen, k=number)
    # parents = random.choices(individuals, k=number)
    return parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    """
    # tournament style elimination
    q: int = 10
    win_amt: dict[int, int] = {}  # individual index : win amount
    for ind_index in range(len(individuals)):
        ind_fit = individuals[ind_index]["fitness"]
        for _ in range(q):
            rand_index: int = random.randrange(len(individuals))
            rand_fit = individuals[rand_index]["fitness"]
            if ind_fit <= rand_fit:
                if ind_index not in win_amt:
                    win_amt[ind_index] = 0
                win_amt[ind_index] += 1

    return_pop: Population = []
    win_sorted: dict[int, int] = {
        k: v for k, v in sorted(win_amt.items(), key=lambda item: item[1], reverse=True)
    }
    for ind_index, win_amt in win_sorted.items():
        return_pop.append(individuals[ind_index])

    return return_pop[:pop_size]


def evolve(io_data: IOdata, pop_size: int = 100) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    recombine_rate: float = .8
    mutate_rate: float = .1
    counter: int = 0
    
    population: Population = initialize_pop(pop_size)
    evaluate_group(population, io_data)
    rank_group(population)

    while population[0]["fitness"] != 13.0:
        parents: Population = parent_select(population, pop_size)        
        children: Population = recombine_group(parents, recombine_rate)
        mutants: Population = mutate_group(children, mutate_rate)
        evaluate_group(mutants, io_data)
        everyone: Population = population + mutants
        rank_group(everyone)
        population = survivor_select(everyone, pop_size)
        # print(counter)
        if counter % 25 == 0:
            parse_tree_print(population[0]["genome"])
            print(f" {population[0]["fitness"]} {counter}")
            # print()
        counter += 1

    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = True

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    print(divider)
    print("Number of possible genetic programs: infinite...")
    print("Lower fitness is better.")
    print(divider)

    X = list(range(-10, 110, 10))
    Y = [(x * (9 / 5)) + 32 for x in X]
    # data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]
    # mypy wanted this:
    data = [initialize_data(input1=x, output1=y) for x, y in zip(X, Y)]

    # Correct:
    print("Example of celcius to farenheight:")
    ind1 = initialize_individual("( + ( * x ( / 9 5 ) ) 32 )", 0)
    evaluate_individual(ind1, data)
    print_tree(ind1["genome"])
    print("Fitness", ind1["fitness"])

    # Yours
    train = data[: int(len(data) / 2)]
    test = data[int(len(data) / 2) :]
    population = evolve(train, POP_SIZE)
    evaluate_individual(population[0], test)
    population[0]["fitness"]

    print("Here is the best program:")
    parse_tree_print(population[0]["genome"])
    print("And it's fitness:")
    print(population[0]["fitness"])
# <<<< DO NOT MODIFY
