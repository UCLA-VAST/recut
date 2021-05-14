import os
from copy import deepcopy
from collections import defaultdict
import numpy as np
from pipeline_util import timer
from .distribution import Distribution
from .neuron_tree import NeuronTree
from .topology import TopologicalGraph
from .graph import BFS, Dijkstra
from .linear_programming import LPSolver


class GCut:
    def __init__(self):
        # tree graph with topological and non topological nodes as vertices,
        # child -> parent connection as edges
        self.tree_nodes = NeuronTree()
        # ids of soma nodes of the system
        self.soma_ids = set()
        # precomputed gof distribution
        self.gof_distribution = Distribution()
        # graph with topological nodes as vertices: soma, branch and leaf and
        # directed neurites as edges
        self.topo_graph = TopologicalGraph()

        self.lp_solver = LPSolver()
        # unordered neurite hash: soma id
        self.neurite_assignments = defaultdict(lambda: None)
        self.neurons = {}
        # tuple(topo_id0, topo_id1), where topo_id0 < topo_id1
        # initialized by copying self.topo_graph.undirected neurites
        self._unassigned_neurites = set()
        self._unassigned_stem_neurites = set()
        self._bfs = BFS()
        self._dijkstra = Dijkstra()

    def clear(self):
        self.tree_nodes.clear()
        self.soma_ids.clear()
        self.gof_distribution.clear()
        self.topo_graph.clear()
        self.lp_solver.clear()
        self.neurite_assignments.clear()
        self._unassigned_neurites.clear()
        self._bfs.clear()
        self._dijkstra.clear()

    # (1) take as input a TreeNode instance
    # (2) create the topological graph, with topological nodes as vertices and
    #     directed neurites as edges
    # (3) gather the set of undirected neurites
    @timer
    def init(self, neuron_tree, soma_ids):
        self.clear()
        assert isinstance(neuron_tree, NeuronTree)
        self.tree_nodes = neuron_tree
        for soma_id in soma_ids:
            assert soma_id in self.tree_nodes.tree
        self.soma_ids = set(soma_ids)
        self.topo_graph.init_topo_nodes(self.tree_nodes, self.soma_ids)
        for soma_id in self.soma_ids:
            assert self.topo_graph.is_soma_node(soma_id)
        self.topo_graph.construct_directed_graph(self.tree_nodes)

    # this function should achieve assignment of all neurites to one and only
    # one neuron
    @timer
    def assign(self, features='gof', conditional='mouse', brain_region='neocortex',
               species='mouse', cell_type='principal neuron'):
        self.gof_distribution.load_distribution(conditional=conditional,
                                                brain_region=brain_region,
                                                species=species, cell_type=cell_type)
        self._unassigned_neurites = deepcopy(self.topo_graph.undirected_hashes())
        self._unassigned_stem_neurites = deepcopy(self.topo_graph.stems())
        # take a random stem neurite, use bfs to find soma nodes and other
        # neurites connected to it. assign soma_id to neurites within
        # this bfs unit
        while len(self._unassigned_neurites) > 0:
            print('remaining unassigned neurites: ', len(self._unassigned_neurites))
            # pick a random stem neurite
            unassigned_stem_neurite = self._unassigned_stem_neurites.pop()
            if unassigned_stem_neurite not in self._unassigned_neurites:
                continue
            self._bfs.clear()
            self._bfs.bfs_neurite(unassigned_stem_neurite, self.topo_graph)
            # the soma set of this computation unit
            unit_soma_ids = self._bfs.connected_topo_ids & self.soma_ids
            assert len(unit_soma_ids) > 0
            # if only one soma is reached by the neurite, the assignment is
            # finished for all neurites reached by the neurite
            if len(unit_soma_ids) == 1:
                print('single soma unit')
                unit_soma_id = unit_soma_ids.pop()
                for neurite in self._bfs.connected_neurites:
                    self.neurite_assignments[neurite] = unit_soma_id
            # otherwise, find neurite assignments by linear programming
            else:
                self.gof_assign(unit_soma_ids)
            self._unassigned_neurites -= self._bfs.connected_neurites
            self._unassigned_stem_neurites -= self._bfs.connected_neurites

    @staticmethod
    def _coefficient_id(nrows, ncols, row, col):
        assert 0 <= row < nrows
        assert 0 <= col < ncols
        return row * ncols + col

    def gof_assign(self, unit_soma_ids):
        print('{} soma unit. gof assign'.format(unit_soma_ids))

        directed_neurite_ids = self.topo_graph.directed_neurite_ids(self._bfs.connected_neurites)
        # compute required gof values for this computation unit
        self.compute_unit_gofs(directed_neurite_ids, unit_soma_ids)

        # construct linear programming optimization constraints
        # the set of neurites considered by the optimization are
        # freeway neurites and ramp entrance neurites. all off ramp neurites
        # assignment are determined when ramp neurite assignment is determined.
        # use the total cost of off ramps + ramp neurite for the ramp neurite
        # in the coefficient matrix.
        # This can be thought of as given a ramp's assignment to soma s,
        # probability of its off ramp assignments to s is equal to that of
        # P(ramp is assigned to s). therefore the ramp and off ramp neurites
        # have identical degree of membership to somas, aka sum of their
        # weighted costs are equal to the weighted sum of their costs
        n_decision_neurites = len(self._bfs.decision_neurites())
        n_somas = len(unit_soma_ids)
        n_coefficients = n_somas * n_decision_neurites
        print('unit size: {} somas, {} undirected neurites'
              .format(n_somas, n_decision_neurites, n_coefficients))

        # imagine the coefficients layout as a matrix C of dimensions
        # [n_somas, n_decisions_neurites]. C[soma, neurite] is the gof cost of
        # the neurite given the soma, with its value determined by Dijkstra
        # algorithm. Each neurite, unless the neurite is predetermined to
        # contain a leaf node, has two possible orientations and therefore
        # two possible gof cost given a soma.
        # sort the undirected neurite hash and soma in ascending orders to track
        # variables properly
        C = np.full((n_somas, n_decision_neurites), fill_value=np.inf)
        self.lp_solver.clear()
        # create variables. all in bounds [0, 1]
        self.lp_solver.create_variables(n_coefficients,
                                        np.zeros((n_coefficients)),
                                        np.ones((n_coefficients,)))
        # create n_dicision_neurites of the equality constraint:
        # sum_{soma_id}(w(soma_id, neurite_id)) = 1
        # membership of each neurite to all somas in the unit must sum to one
        for j in np.arange(n_decision_neurites):
            variable_indices = np.arange(j, n_coefficients, n_decision_neurites)
            constraint_coefficients = np.ones((n_somas,))
            self.lp_solver.add_constraint(variable_indices, constraint_coefficients, 1, 1)

        # determine the row and column number for each soma and unoriented
        # neurite in the coefficients matrix
        ordered_neurites = TopologicalGraph.sort_undirected_neurites(self._bfs.decision_neurites())
        neurite_col = {}
        for i, neurite_hash in enumerate(ordered_neurites):
            neurite_col[neurite_hash] = i
        ordered_somas = sorted(list(unit_soma_ids))
        soma_row = {}
        for i, soma_id in enumerate(ordered_somas):
            soma_row[soma_id] = i

        for unit_soma_id in unit_soma_ids:
            self._dijkstra.clear()
            self._dijkstra.dijkstra(unit_soma_id, self._bfs.connected_topo_ids, self.topo_graph)
            # paris of (child undirected neurite, parent undirected neurite)
            # each neurite is in the form of DirectedNeurite.undirected_hash()
            neurite_paternity = self._dijkstra.neurite_lineages(self._bfs.ramps, neurite_subset=self._bfs.decision_neurites())
            # w(soma_id, neurite_id) <= w(soma_id, parent(neurite_id, soma_id))
            # this constraint is constructed from Dijkstra result
            # there are (n_decision_neurites - 1) * n_somas of such constraints
            # for each (child undirected neurite, parent undirected neurite)
            # construct an inequality constraint: child - parent <= 0
            # since variables are all within [0, 1], there should also be
            # child - parent >= -1
            for (neurite, parent) in neurite_paternity:
                row, col, col_parent = soma_row[unit_soma_id], neurite_col[neurite], neurite_col[parent]
                index = GCut._coefficient_id(n_somas, n_decision_neurites, row, col)
                parent_index = GCut._coefficient_id(n_somas, n_decision_neurites, row, col_parent)
                self.lp_solver.add_constraint([index, parent_index], [1, -1], -1, 0)
            # fill coefficients values with dijkstra edge cost result
            decision_neurite_costs = self._dijkstra.decision_neurite_costs(self._bfs.ramps)
            columns, costs = [], []
            # use the Dijkstra results to update coefficient matrix
            for neurite_hash, cost in decision_neurite_costs.items():
                columns.append(neurite_col[neurite_hash])
                costs.append(cost)
            C[[soma_row[unit_soma_id]] * n_decision_neurites, columns] = costs
        self.lp_solver.define_objective(C.ravel())
        print('linear programming: n coefficients = {}, n constraints = {}'
              .format(self.lp_solver.n_variables(),
                      self.lp_solver.n_constraints()))
        self.lp_solver.solve()

        # neurite membership to each soma in this unit
        memberships = self.lp_solver.solutions().reshape((n_somas, n_decision_neurites))
        optimal_soma_row = np.argmax(memberships, axis=0)
        # assign decision neurite to the soma to which it has the greatest
        # degree of membership
        for i in range(n_decision_neurites):
            self.neurite_assignments[ordered_neurites[i]] = ordered_somas[optimal_soma_row[i]]
        # assign off ramp neurites to same soma as ramp neurite
        for ramp_neurite, off_freeway_neurites in self._bfs.ramps.items():
            for off_freeway_neurite in off_freeway_neurites:
                self.neurite_assignments[off_freeway_neurite] = self.neurite_assignments[ramp_neurite]

    @timer
    def assemble(self):
        assert len(self._unassigned_neurites) == 0
        assignments = {}
        for neurite in self.neurite_assignments:
            assignments[str(neurite)] = self.neurite_assignments[neurite]
        for soma_id in self.soma_ids:
            self.assemble_one(soma_id)
        return self.neurons

    def assemble_one(self, soma_id):
        assert soma_id in self.soma_ids
        self.neurons[soma_id] = NeuronTree()
        soma_node = deepcopy(self.tree_nodes.tree[soma_id])
        soma_node.parent_id = NeuronTree.ROOT_PARENT_ID
        self.neurons[soma_id].add_node(soma_node)
        neurite_hashes = set([neurite_hash for neurite_hash, soma in
                              self.neurite_assignments.items() if soma == soma_id])
        tip_ids = [soma_id]
        # direction of neurite is tip -> tip neighbor when starting tip is soma
        while len(tip_ids) > 0:
            tip_id = tip_ids.pop(0)
            topo_target_ids = [topo_id for topo_id in self.topo_graph.graph[tip_id].out_neurites.keys()]
            for topo_target_id in topo_target_ids:
                neurite_hash = TopologicalGraph.undirected_hash(tip_id, topo_target_id)
                if neurite_hash not in neurite_hashes:
                    continue
                neurite_hashes.remove(neurite_hash)
                growth_neurite_id = self.topo_graph.graph[tip_id].out_neurites[topo_target_id]
                growth_node_ids = self.topo_graph.neurites[growth_neurite_id].node_ids
                for i in range(1, len(growth_node_ids)):
                    growth_node = deepcopy(self.tree_nodes.tree[growth_node_ids[i]])
                    growth_node.parent_id = growth_node_ids[i - 1]
                    self.neurons[soma_id].add_node(growth_node)
                tip_ids.append(topo_target_id)
        # let the assembled neuron inherit the metalines from input neuron tree
        self.neurons[soma_id].write_to_meta_lines(self.tree_nodes.meta_lines)
        # log operation in assembled neuron metaline
        self.neurons[soma_id].log_operation()

    def compute_unit_gofs(self, neurite_ids, soma_ids):
        self._compute_gofs(neurite_ids, soma_ids)
        self._compute_gof_probability(neurite_ids)

    @timer
    def _compute_gofs(self, neurite_ids, soma_ids):
        for neurite_id in neurite_ids:
            self.topo_graph.neurites[neurite_id].compute_gof(soma_ids, self.tree_nodes)

    @timer
    def _compute_gof_probability(self, neurite_ids):
        for neurite_id in neurite_ids:
            self.topo_graph.neurites[neurite_id].compute_gof_probability(self.gof_distribution)

    def neurites_to_swcs(self, out_dir=""):
        if len(self.topo_graph.neurites) ==0:
            print('no neurites. do nothing')
            return
        #os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'neurites.swc'),'w') as f:
            new_node_id = 1
            for neurite_id, neurite in self.topo_graph.neurites.items():
                for i, node_id in enumerate(neurite.node_ids):
                    node = self.tree_nodes.tree[node_id]
                    parent_id = new_node_id - 1 if i > 0 else NeuronTree.ROOT_PARENT_ID
                    f.write('{} {} {} {} {} {} {}\n'
                            .format(new_node_id, node.node_type,
                                    node.x, node.y, node.z,
                                    node.radius, parent_id))
                    new_node_id += 1

