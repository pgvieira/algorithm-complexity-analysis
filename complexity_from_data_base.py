import javalang
import numpy as np
import pandas as pd
import os.path as op
from javalang import tree


complexity_dictionary_binomial = {'n': 0, 'n_square': 1, '1': 0, 'nlogn': 1, 'logn': 0}
complexity_dictionary_multinomial = {'n': 0, 'n_square': 1, '1': 2, 'nlogn': 3, 'logn': 4}

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map', 'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']

data_binomial = pd.DataFrame(columns=array_suport)
data_multinomial = pd.DataFrame(columns=array_suport)


def state_counter(typelist):
    num_state = 0
    num_state += typelist.count(javalang.tree.IfStatement)
    num_state += typelist.count(javalang.tree.WhileStatement)
    num_state += typelist.count(javalang.tree.DoStatement)
    num_state += typelist.count(javalang.tree.ForStatement)
    num_state += typelist.count(javalang.tree.AssertStatement)
    num_state += typelist.count(javalang.tree.BreakStatement)
    num_state += typelist.count(javalang.tree.ContinueStatement)
    num_state += typelist.count(javalang.tree.ReturnStatement)
    num_state += typelist.count(javalang.tree.ThrowStatement)
    num_state += typelist.count(javalang.tree.SynchronizedStatement)
    num_state += typelist.count(javalang.tree.TryStatement)
    num_state += typelist.count(javalang.tree.SwitchStatement)
    num_state += typelist.count(javalang.tree.BlockStatement)
    return num_state


def extract_value_from_file(string_file):
    path_to_file = 'base_dados/code_java/' + string_file

    if op.isfile(path_to_file):
        file = open(path_to_file)
        source = file.read()
        file.close()

        try:
            tree = javalang.parse.parse(source)
        except:
            return []

        typelist = []

        num_recursive = 0
        num_nasted_loop = 0
        num_hash_map = 0
        num_hash_set = 0
        num_Priority = 0
        num_sort = 0

        for path, node in tree:
            typelist.append(type(node))
            if type(node) == javalang.tree.MethodDeclaration:
                temp_name = node.name
                if node.name in str(node).replace('name=' + temp_name, ''):
                    num_recursive += 1
            if type(node) == javalang.tree.ForStatement:
                if str(node).count('ForStatement') != 1 and str(node).count('ForStatement') > num_nasted_loop:
                    num_nasted_loop = str(node).count('ForStatement')
            if type(node) == javalang.tree.LocalVariableDeclaration:
                try:
                    if node.declarators[0].initializer.type.name == 'HashMap':
                        num_hash_map += 1
                    elif node.declarators[0].initializer.type.name == 'HashSet':
                        num_hash_set += 1
                    elif node.declarators[0].initializer.type.name == 'PriorityQueue':
                        num_Priority += 1
                except:
                    pass
            if type(node) == javalang.tree.MethodInvocation:
                if node.member == 'sort':
                    num_sort += 1

        num_if = typelist.count(javalang.tree.IfStatement)
        num_loof = typelist.count(javalang.tree.ForStatement) + typelist.count(javalang.tree.WhileStatement)
        num_vari = typelist.count(javalang.tree.VariableDeclaration) + typelist.count(javalang.tree.LocalVariableDeclaration)
        num_break = typelist.count(javalang.tree.BreakStatement)
        num_method = typelist.count(javalang.tree.MethodDeclaration)
        num_switch = typelist.count(javalang.tree.SwitchStatement)
        num_state = state_counter(typelist)

        return [num_if, num_switch, num_loof, num_break, num_Priority, num_sort, num_hash_map, num_hash_set, num_recursive, num_nasted_loop, num_vari, num_method, num_state]
    else:
        return []


data_frame = pd.read_csv('base_dados/base_path_file_complexity.csv').dropna()
data_name = data_frame['file_name'].reset_index(drop=True)
data_complexity = data_frame['complexity'].reset_index(drop=True)

data_complexity_binomial = np.array([int(complexity_dictionary_binomial[row]) for row in data_complexity])
data_complexity_multinomial = np.array([int(complexity_dictionary_multinomial[row]) for row in data_complexity])

for num, val in enumerate(data_name):
    values_from_file = extract_value_from_file(val)
    if values_from_file:
        data_binomial.loc[num] = values_from_file
        data_multinomial.loc[num] = values_from_file


data_binomial = data_binomial.assign(complexity=data_complexity_binomial)
data_multinomial = data_multinomial.assign(complexity=data_complexity_multinomial)

data_binomial.to_csv('base_dados/out_binomial.csv', index=False)
data_multinomial.to_csv('base_dados/out_multinomial.csv', index=False)
