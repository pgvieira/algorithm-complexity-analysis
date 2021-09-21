import javalang

import numpy as np
import os.path as op
import pandas as pd
from javalang import tree


complexity_dictionary = {'n': 0, 'n_square': 1, '1': 2, 'nlogn': 3, 'logn': 4}

array_suport = ['num_If', 'num_Switch', 'num_Loof', 'num_Break', 'num_Priority', 'num_Sort', 'num_Hash_Map',
                'num_Hash_Set', 'num_Recursive', 'num_Nasted_Loop', 'num_Variables', 'num_Method', 'num_State']

data = pd.DataFrame(columns=array_suport)


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
    path_to_file = path_base_data + '/' + string_file

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
        num_vari = typelist.count(javalang.tree.VariableDeclaration) + typelist.count(
            javalang.tree.LocalVariableDeclaration)
        num_break = typelist.count(javalang.tree.BreakStatement)
        num_method = typelist.count(javalang.tree.MethodDeclaration)
        num_switch = typelist.count(javalang.tree.SwitchStatement)
        num_state = state_counter(typelist)

        return [num_if, num_switch, num_loof, num_break, num_Priority, num_sort, num_hash_map, num_hash_set,
                num_recursive, num_nasted_loop, num_vari, num_method, num_state]
    else:
        return []


array_suport = np.zeros(13)

path_base_data = 'base_dados/code_java'

temp_data = pd.read_csv('base_dados/finalFeatureData.csv').dropna()
data_name = temp_data['file_name'].reset_index(drop=True)
data_complexity = temp_data['complexity'].reset_index(drop=True)

data_complexity = np.array([int(complexity_dictionary[row]) for row in data_complexity])

for num, i in enumerate(data_name):
    values_from_file = extract_value_from_file(i)
    if values_from_file:
        data.loc[num] = values_from_file


data = data.assign(complexity=data_complexity)

data.to_csv('out.csv', index=False)



