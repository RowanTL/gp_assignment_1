from gp_lisp import parse_expression, Node, get_nodes, put_an_x_in_it_node, parse_tree_print

def main() -> None:
    exp_str: str = "( * ( + 20 45 ) ( - 56 x ) )"
    exp_node: Node = parse_expression(exp_str)
    parse_tree_print(exp_node)
    print()

    put_an_x_in_it_node(exp_node)
    parse_tree_print(exp_node)
    print()
    return


if __name__ == "__main__":
    main()