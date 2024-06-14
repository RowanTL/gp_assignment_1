from gp_lisp import parse_expression, Node, get_nodes, put_an_x_in_it_node, parse_tree_print, recombine_pair, initialize_individual, mutate_individual

def main() -> None:
    exp0_str: str = "( * ( + 20 45 ) ( - 56 x ) )"
    # exp0_node: Node = parse_expression(exp0_str)
    exp1_str: str = "( / ( - 90 x ) ( + 3 6 ) )"
    # exp1_node: Node = parse_expression(exp1_str)
    ind0 = initialize_individual(exp0_str, 0.0)
    exp_bad_str: str = '( - -( * -46 x ) ( * -46 x ) )'
    exp_bad_node: Node = parse_expression(exp_bad_str)
    # ind1 = initialize_individual(exp1_str, 0.0)
    # parse_tree_print(exp_node)
    # print()

    # put_an_x_in_it_node(exp_node)
    # parse_tree_print(exp_node)
    # print()
    # nodes: list[Node] = get_nodes(exp_node)
    # for node in nodes:
        # print(node.data)

    # c0, c1 = recombine_pair(ind0, ind1)
    # print()
    # parse_tree_print(c0["genome"])
    # print()
    # parse_tree_print(c1["genome"])

    # m1 = mutate_individual(ind0, 1.0)

    return


if __name__ == "__main__":
    main()
