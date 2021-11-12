from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class Tree :
    tag = ''
    leff_arc = None
    right_arc = None
    isLeaf = False

    def __init__(self, tag, la, ra, isLeaf) :
        self.tag = tag
        self.leff_arc = la
        self.right_arc = ra
        self.isLeaf = isLeaf

    def toString (self):
        list = []
        left_s = str(self.tag).strip('[]').strip('\'')
        list.append(left_s)
        list.append('(')
        middle_s = None

        if self.isLeaf:
            middle_s = str(self.leff_arc).strip('[]').strip('\'')
        else:
            list2 = []
            list2.append(self.leff_arc.toString())
            list2.append(self.right_arc.toString())
            middle_s = ', '.join(map(str, list2))

        list.append(middle_s)
        list.append(')')

        return ''.join(map(str, list))
        
class CKY :

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []

    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file) :
        stream = open( grammar_file, mode='r', encoding='utf8' )
        for line in stream :
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2 :
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules :
                    first_rules = self.binary_rules[first]
                else :
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules :
                    second_rules = first_rules[second]
                    if left not in second_rules :
                        second_rules.append[left]
                else :
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1 :
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules :
                    word_rules = self.unary_rules[word]
                    if left not in word_rules :
                        word_rules.append( left )
                else :
                    word_rules = [left]
                    self.unary_rules[word] = word_rules


    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s) :
        self.words = s.split()    
        n = len(self.words)

        # Initialize the table and back pointers
        for _ in range(n):
            row = []
            tree_row = []
            for _ in range(n):
                row.append([])
                tree_row.append([])
            self.table.append(row)
            self.backptr.append(tree_row)

        # Compute value for every cell in the table columnwise
        # Column: left -> right
        # Row: bottom -> top
        for col in range(n):
            word = self.words[col]
            for row in range(n-1, -1, -1):

                # Cell is on diagonal
                if (col == row):
                    tag = self.unary_rules[word]
                    self.table[row][col] = tag

                    for t in tag:
                        leaf = Tree (t, word, None, True)
                        self.backptr[row][col].append(leaf)

                # Cell is above diagonal
                elif (col > row):
                    # combind all cells from the same row with cells from the same column
                    for sub_col in range(col):
                        sub_row = sub_col + 1
                        self.table_append(sub_col, sub_row, row, col)

                        # if col == 3 and row == 0:
                        #     print('(',row, ',', sub_col, ')', ' + ', '(', sub_row , ',', col, ')')


    def table_append (self, sub_col, sub_row, row, col):
        list1 = self.table[row][sub_col]
        list2 = self.table[sub_row][col]
        #print('row: ', row, 'col: ', col)
        for i, tag1 in enumerate(list1):
            for j, tag2 in enumerate(list2): 
                if tag1 in self.binary_rules:
                    nextElem = self.binary_rules[tag1]
                    if tag2 in nextElem:
                        tag = nextElem[tag2]
                        self.table[row][col].append(tag[0])
                        
                        left_tree = self.backptr[row][sub_col][i]
                        #print(sub_row, ', ', col, ', ', j)
                        right_tree = self.backptr[sub_row][col][j]
                        tree = Tree (tag, left_tree, right_tree, False)
                        self.backptr[row][col].append(tree)

    # Prints the parse table
    def print_table( self ) :
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print( t.table )


    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees( self, row, column, symbol ) :
        print()
        print('Trees found with root \'', symbol, '\' :')
        list = self.table[row][column]
        for i,l in enumerate(list):
            if(l == symbol):
                tree = self.backptr[row][column][i]
                print(tree.toString()) 


def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str,  required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY( arguments.grammar )
    cky.parse( arguments.input_sentence )
    if arguments.print_parsetable :
        cky.print_table()
    if arguments.print_trees :
        cky.print_trees( 0, len(cky.words)-1, arguments.symbol )
    
if __name__ == '__main__' :
    main()    


                        
                        
                        
                    
                
                    

                
        
    
