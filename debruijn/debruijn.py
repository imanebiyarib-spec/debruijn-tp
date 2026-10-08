#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random
import networkx as nx

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Imane"
__copyright__ = "Universite Paris Cité"
__credits__ = ["Imane"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Imane"
__email__ = "@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as f:
        for line in f: 
            read = next(f).strip() 
            next(f) # + line 
            next(f) # quality line
            yield read 


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file): # pick each seq from read_fastq
        for kmer in cut_kmer(read, kmer_size): # each kmers for this seq
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1 
            
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    
    graph = DiGraph() # directed graph (library networkx)
    
    for kmer, count in kmer_dict.items(): #each kmer and its occurrence 
        prefix = kmer[:-1]
        suffix = kmer[1:]
        
        graph.add_edge(prefix, suffix, weight=count)
        
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
            start_idx = 0 if delete_entry_node else 1
            end_idx = None if delete_sink_node else -1
            
            nodes_to_remove = path[start_idx:end_idx]
            graph.remove_nodes_from(nodes_to_remove)
            
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    best_idx = 0
    
    if len(path_list) > 1:
        #  difference in average weights ?
        if statistics.stdev(weight_avg_list) > 0:
            best_idx = weight_avg_list.index(max(weight_avg_list))
        # weights are identical,  difference in lengths?
        elif statistics.stdev(path_length) > 0:
            best_idx = path_length.index(max(path_length))
        # everything is identical, pick randomly
        else:
            best_idx = randint(0, len(path_list) - 1)

    # Filter out the best path 
    paths_to_remove = [
        path for i, path in enumerate(path_list) if i != best_idx
    ]

    # delete other paths
    return remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """

    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    lengths = [len(p) for p in paths]
    weights = [path_average_weight(graph, p) for p in paths]

    return select_best_path(
        graph, paths, lengths, weights, delete_entry_node=False, delete_sink_node=False
    )


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble_found = False
    ancestor = None
    descendant = None

    for node in list(graph.nodes()):
        preds = list(graph.predecessors(node))
        
        if len(preds) > 1: # end of a bubble
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    anc = lowest_common_ancestor(graph, preds[i], preds[j])
                    
                    if anc is not None:
                        bubble_found = True
                        ancestor = anc
                        descendant = node
                        break # Exit inner loop
                    
                if bubble_found:
                    break # Exit middle loop
                
        if bubble_found:
            break # Exit outer loop

    # bubble found, we solve it -->  changes the graph structure
    if bubble_found:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, descendant))

    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    tip_found = False
    convergence_node = None
    involved_starts = []
    
    # Find a convergence node
    for node in list(graph.nodes()):
        if len(list(graph.predecessors(node))) > 1:
            connected_starts = [s for s in starting_nodes if has_path(graph, s, node)]
            if len(connected_starts) > 1:
                tip_found = True
                convergence_node = node
                involved_starts = connected_starts
                break
                
    # Resolve the detected tip
    if tip_found:
        paths = []
        for start in involved_starts:
            paths.extend(list(all_simple_paths(graph, start, convergence_node)))
        
        lengths = [len(p) for p in paths]
        weights = [path_average_weight(graph, p) for p in paths]
        
        # Remove the entry node of the tip (True) but keep the convergence node (False)
        graph = select_best_path(
            graph, paths, lengths, weights, 
            delete_entry_node=True, delete_sink_node=False
        )
        # Recursive call with updated starting nodes
        graph = solve_entry_tips(graph, get_starting_nodes(graph))
        
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    tip_found = False
    divergence_node = None
    involved_ends = []
    
    # Find a divergence node
    for node in list(graph.nodes()):
        if len(list(graph.successors(node))) > 1:
            connected_ends = [e for e in ending_nodes if has_path(graph, node, e)]
            if len(connected_ends) > 1:
                tip_found = True
                divergence_node = node
                involved_ends = connected_ends
                break
                
    # Resolve the detected tip
    if tip_found:
        paths = []
        for end in involved_ends:
            paths.extend(list(all_simple_paths(graph, divergence_node, end)))
            
        lengths = [len(p) for p in paths]
        weights = [path_average_weight(graph, p) for p in paths]
        
        graph = select_best_path(
            graph, paths, lengths, weights, 
            delete_entry_node=False, delete_sink_node=True
        )

        # updated sink nodes
        graph = solve_out_tips(graph, get_sink_nodes(graph))
        
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes():
        if len(list(graph.predecessors(node))) == 0: # no entry
            starting_nodes.append(node)
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    ending_nodes = []
    for node in graph.nodes():
        if len(list(graph.successors(node))) == 0: # no exit
            ending_nodes.append(node)
    return ending_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """

    contigs = []

    for start in starting_nodes:
        for end in ending_nodes:

            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    contig = path[0] # start the sequence

                    for node in path[1:]:
                        contig += node[-1] # add nucleotide
                    
                    contigs.append((contig, len(contig)))
                    
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as f:
            for i, (contig, length) in enumerate(contigs_list):
                f.write(f">contig_{i} len={length}\n") # header
                f.write(textwrap.fill(contig, width=80) + "\n") # sequence (max 80 nc/line)


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Read file and build graph
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)

    # Resolve bubbles
    graph = simplify_bubbles(graph)

    # Resolve entry and out tips
    start_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, start_nodes)

    end_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, end_nodes)

    # Write contigs
    final_start_nodes = get_starting_nodes(graph)
    final_end_nodes = get_sink_nodes(graph)
    
    contigs = get_contigs(graph, final_start_nodes, final_end_nodes)
    save_contigs(contigs, args.output_file)

    #  draw the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
