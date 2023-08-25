'''
parseidf.py


parses an idf file into a dictionary of lists in the following manner:

    each idf object is represented by a list of its fields, with the first
    field being the objects type.

    each such list is appended to a list of objects with the same type in the
    dictionary, indexed by type:

    { [A] => [ [A, x, y, z], [A, a, b, c],
      [B] => [ [B, 1, 2], [B, 1, 2, 3] }

    also, all field values are strings, i.e. no interpretation of the values is
    made.
'''
import ply.lex as lex
import ply.yacc as yacc

from typing import List

import pprint
import json # json.dumps is used for pprinting dictionaries

import networkx as nx
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

# list of token names
tokens = ('VALUE',
          'COMMA',
          'SEMICOLON')

# regular expression rules for simple tokens
t_COMMA = r'[ \t]*,[ \t]*'
t_SEMICOLON = r'[ \t]*;[ \t]*'


# ignore comments, tracking line numbers at the same time
def t_COMMENT(t):
    r'[ \t\r\n]*!.*'
    newlines = [n for n in t.value if n == '\n']
    t.lineno += len(newlines)
    pass
    # No return value. Token discarded


# Define a rule so we can track line numbers
def t_newline(t):
    r'[ \t]*(\r?\n)+'
    t.lexer.lineno += len(t.value)


def t_VALUE(t):
    r'[ \t]*([^!,;\n]|[*])+[ \t]*'
    return t


# Error handling rule
def t_error(t):
    raise SyntaxError("Illegal character '%s' at line %d of input"
                      % (t.value[0], t.lexer.lineno))
    t.lexer.skip(1)


# define grammar of idf objects
def p_idffile(p):
    'idffile : idfobjectlist'
    result = {}
    for idfobject in p[1]:
        name = idfobject[0]
        result.setdefault(name.upper(), []).append(idfobject)
    p[0] = result


def p_idfobjectlist(p):
    'idfobjectlist : idfobject'
    p[0] = [p[1]]


def p_idfobjectlist_multiple(p):
    'idfobjectlist : idfobject idfobjectlist'
    p[0] = [p[1]] + p[2]


def p_idfobject(p):
    'idfobject : objectname SEMICOLON'
    p[0] = [p[1]]


def p_idfobject_with_values(p):
    'idfobject : objectname COMMA valuelist SEMICOLON'
    p[0] = [p[1]] + p[3]


def p_objectname(p):
    'objectname : VALUE'
    p[0] = p[1].strip()


def p_valuelist(p):
    'valuelist : VALUE'
    p[0] = [p[1].strip()]


def p_valuelist_multiple(p):
    'valuelist : VALUE COMMA valuelist'
    p[0] = [p[1].strip()] + p[3]


# oh, and handle errors
def p_error(p):
    raise SyntaxError("Syntax error in input on line %d" % lex.lexer.lineno)


def parse(input) -> dict:
    '''
    parses a string with the contents of the idf file and returns the
    dictionary representation.
    '''
    lexer = lex.lex(debug=False)
    lexer.input(input)
    parser = yacc.yacc()
    result = parser.parse(debug=False)
    return result

def get_zone_list(parsed_idf):
    ZONE_NAME = 4
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']
    surfaces_set = set()
    for building_surface in building_surfaces:
        surfaces_set.add(building_surface[ZONE_NAME])
    return list(surfaces_set)

def get_solar_surface_list(parsed_idf):
    '''
    get surfaces + windows
    '''
    B_SOLAR = 8
    NAME = 1
    ret_list = []
    for window in parsed_idf['WINDOW']:
        ret_list.append(window[NAME])
    for surface in parsed_idf['BUILDINGSURFACE:DETAILED']:
        if surface[B_SOLAR] == 'SunExposed':
            ret_list.append(surface[NAME])
    return ret_list

def get_surface_to_zone_dict(parsed_idf) -> dict:
    '''
    key: surface_name
    val: zone that the surface is in
    '''
    OUTSIDE_BOUNDARY_CONDITION = 6
    NAME = 1
    ZONE_NAME = 4

    ret_dict = dict()
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']

    # filter ignorable surfaces (eg.Adiabatic)
    for building_surface in building_surfaces:
        if building_surface[OUTSIDE_BOUNDARY_CONDITION].upper() not in ['SURFACE', 'ZONE', 'OUTDOORS', 'GROUND']:
            continue
        else:
            ret_dict[building_surface[NAME]] = building_surface[ZONE_NAME]
    return ret_dict

def get_surface_connect_surface(parsed_idf) -> dict:
    '''
    key: surface name
    val: get zone the surface is connected to
    '''
    OUTSIDE_BOUNDARY_CONDITION = 6
    OUTSIDE_BOUNDARY_CONDITION_OBJECT = 7
    NAME = 1
    ZONE_NAME = 4

    ret_dict = dict()
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']

    for building_surface in building_surfaces:
        boundary_condition = building_surface[OUTSIDE_BOUNDARY_CONDITION]
        if boundary_condition == "Surface":
            ret_dict[building_surface[NAME]] = building_surface[OUTSIDE_BOUNDARY_CONDITION_OBJECT]
        elif boundary_condition == "Ground":
            ret_dict[building_surface[NAME]] = "Ground"
        elif boundary_condition == "Outdoors":
            ret_dict[building_surface[NAME]] = "Outdoors"
        elif boundary_condition == "Zone":
            ret_dict[building_surface[NAME]] = building_surface[OUTSIDE_BOUNDARY_CONDITION_OBJECT]
        else:
            # e.g. adiabatic
            continue

    return ret_dict

def directed_zone_connections(parsed_idf) -> list:
    '''
    @param: parsed_idf - idf file parsed into a dict
    NOTE:
    - outside boundary condition:

    connections of zones. Converting the outputted list will result in a directed graph
    of the zone connection dynamics
    '''
    # below is const for BuildingSurface:Detailed IDF obj
    BUILDING_SURFACE_DETAILED = 0
    NAME = 1
    SURFACE_TYPE = 2
    CONSTRUCTION_NAME = 3
    ZONE_NAME = 4
    SPACE_NAME = 5
    OUTSIDE_BOUNDARY_CONDITION = 6
    OUTSIDE_BOUNDARY_CONDITION_OBJECT = 7
    SUN_EXPOSURE = 8
    WIND_EXPOSURE = 9
    VIEW_FACTOR_TO_GROUND = 10
    NUMBER_OF_VERTICES = 11
    # 12 ~ 23 vertices X,Y,Z stuff

    # get surface_to_zone dict
    surface_to_zone = get_surface_to_zone_dict(parsed_idf)

    # get surface_to_surface connection dict
    surface_connect_surface = get_surface_connect_surface(parsed_idf)

    # get zone_list
    zone_list = get_zone_list(parsed_idf)

    zone_connection_dict = dict()
    zone_connection_list = []
    for surface in surface_to_zone.keys():
        start_zone = surface_to_zone[surface]

        connected_surface = surface_connect_surface[surface]
        # case 1: connected_surface is Ground
        # case 2: connected_surface is Zone name
        # case 3: connecte_surface is Outdoors
        if connected_surface == "Outdoors":
            #zone_connection_dict[start_zone] = 'Outdoors'
            zone_connection_list.append([start_zone, 'Outdoors'])
        elif connected_surface == "Ground":
            #zone_connection_dict[start_zone] = 'Ground'
            zone_connection_list.append([start_zone, 'Ground'])
        elif connected_surface in zone_list:
            #zone_connection_dict[start_zone] = connected_surface # it would be "connected_zone" for this case
            zone_connection_list.append([start_zone, connected_surface])
        else:
            end_zone = surface_to_zone[surface_connect_surface[surface]]
            #zone_connection_dict[start_zone] = end_zone
            zone_connection_list.append([start_zone, end_zone])

    return zone_connection_list

def directed_to_undirected_zone(directed_list):
    '''
    @param: directed_list - list that represents the zone connections as a directed graph
    directed_list: List[List(str, str)]
    '''
    undirected_graph = []
    for connection in directed_list:
        start, end = connection
        undirected_graph.append([start, end])
        undirected_graph.append([end, start])

    return list(undirected_graph)

def main(parsed_idf):
    l = directed_zone_connections(parsed_idf)
    return directed_to_undirected_zone(l)

def visualize_connections(connections):
    G = nx.Graph()
    for connection in connections:
        start, end = connection
        G.add_edge(start, end)
    pos = nx.spring_layout(G, seed=42)  # You can use different layout algorithms if needed
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, font_weight="bold")
    plt.show()

def generate_connections(idf_f_path: str):
    idf_file = open(idf_f_path, 'r')
    f = idf_file.read()
    res = parse(f)
    return main(res)

def generate_adjacency(idf_f_path: str):
    edges_list: list = generate_connections(idf_f_path)
    # Create an empty adjacency list dictionary
    adjacency_list = {}

    # Convert the list of edges into an adjacency list dictionary
    for edge in edges_list:
        node1, node2 = edge
        if node1 not in adjacency_list:
            adjacency_list[node1] = []
        if node2 not in adjacency_list:
            adjacency_list[node2] = []

        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)

    for zone in adjacency_list:
        adjacency_list[zone] = list(set(adjacency_list[zone]))

    return adjacency_list

if __name__ == "__main__":
    # idf_file = open('./5ZoneAirCooledConvCoef.idf', 'r')
    idf_file = open('./in.idf', 'r')
    f = idf_file.read()
    res = parse(f)

    #print(res['BUILDINGSURFACE:DETAILED'][0].index('NoSun'))
    print(get_solar_surface_list(res))

    # l2 = main(res)
    # visualize_connections(l2)
    #pp.pprint(l2)

    #print(json.dumps(get_surface_to_zone_dict(res), indent=4))
    # print(res['BUILDINGSURFACE:DETAILED'], type(res['BUILDINGSURFACE:DETAILED']))
    # pp.pprint(get_zone_list(res))
