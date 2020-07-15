#!/usr/bin/env python
# coding: utf-8

import itertools

import numpy as np
import pandas as pd
import plotly
import scipy.spatial
from plotly import express as px
from plotly import graph_objs as go
from umap import UMAP
from tqdm.notebook import tqdm

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from IPython.display import Image

from . import relations as rel


def flatten(iterable):
    return itertools.chain.from_iterable(iterable)


def feature_matrix(ces, relations):
    """Return a matrix representing each cause and effect in the CES.

    .. note::
        Assumes that causes and effects have been separated.
    """
    N = len(ces)
    M = len(relations)
    # Create a mapping from causes and effects to indices in the feature matrix
    index_map = {purview: i for i, purview in enumerate(ces)}
    # Initialize the feature vector
    features = np.zeros([N, M])
    # Assign features
    for j, relation in enumerate(relations):
        indices = [index_map[relatum] for relatum in relation.relata]
        # Create the column corresponding to the relation
        relation_features = np.zeros(N)
        # Assign 1s where the cause/effect purview is involved in the relation
        relation_features[indices] = 1
        # Assign the feature column to the feature matrix
        features[:, j] = relation_features
    return features


def get_coords(data, y=None, **params):
    umap = UMAP(
        n_components=3, metric="euclidean", n_neighbors=30, min_dist=0.5, **params,
    )
    return umap.fit_transform(data, y=y)


def relation_vertex_indices(features, j):
    """Return the indices of the vertices for relation ``j``."""
    return features[:, j].nonzero()[0]


def all_triangles(vertices):
    """Return all triangles within a set of vertices."""
    return itertools.combinations(vertices, 3)


def all_edges(vertices):
    """Return all edges within a set of vertices."""
    return itertools.combinations(vertices, 2)


def make_label(nodes, node_labels=None):
    if node_labels is not None:
        nodes = node_labels.indices2labels(nodes)
    return "".join(nodes)


def label_mechanism(mice):
    return make_label(mice.mechanism, node_labels=mice.node_labels)


def label_purview(mice):
    return make_label(mice.purview, node_labels=mice.node_labels)


def label_state(mice):
    return [rel.maximal_state(mice)[0][node] for node in mice.purview]


def label_relation(relation):
    relata = relation.relata

    relata_info = "<br>".join(
        [
            f"{label_mechanism(mice)} / {label_purview(mice)} [{mice.direction.name}]"
            for n, mice in enumerate(relata)
        ]
    )

    relation_info = f"<br>Relation purview: {make_label(relation.purview, relation.subsystem.node_labels)}<br>Relation φ = {phi_round(relation.phi)}<br>"

    return relata_info + relation_info


def hovertext_mechanism(distinction):
    return f"Distinction: {label_mechanism(distinction.cause)}<br>Cause: {label_purview(distinction.cause)}<br>Cause φ = {phi_round(distinction.cause.phi)}<br>Cause state: {[rel.maximal_state(distinction.cause)[0][i] for i in distinction.cause.purview]}<br>Effect: {label_purview(distinction.effect)}<br>Effect φ = {phi_round(distinction.effect.phi)}<br>Effect state: {[rel.maximal_state(distinction.effect)[0][i] for i in distinction.effect.purview]}"


def hovertext_purview(mice):
    return f"Distinction: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.maximal_state(mice)[0][i] for i in mice.purview]}"


def hovertext_relation(relation):
    relata = relation.relata

    relata_info = "".join(
        [
            f"<br>Distinction {n}: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.maximal_state(mice)[0][i] for i in mice.purview]}<br>"
            for n, mice in enumerate(relata)
        ]
    )

    relation_info = f"<br>Relation purview: {make_label(relation.purview, relation.subsystem.node_labels)}<br>Relation φ = {phi_round(relation.phi)}<br>"

    return f"<br>={len(relata)}-Relation=<br>" + relata_info + relation_info


def normalize_sizes(min_size, max_size, elements):
    phis = np.array([element.phi for element in elements])
    min_phi = phis.min()
    max_phi = phis.max()
    return min_size + (((phis - min_phi) * (max_size - min_size)) / (max_phi - min_phi))


def phi_round(phi):
    return np.round(phi, 4)


def chunk_list(my_list, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(my_list), n):
        yield my_list[i : i + n]


def format_node(n, subsystem):
    node_format = {
        "label": subsystem.node_labels[n],
        "style": "filled" if subsystem.state[n] == 1 else "",
        "fillcolor": "black" if subsystem.state[n] == 1 else "",
        "fontcolor": "white" if subsystem.state[n] == 1 else "black",
    }
    return node_format


def save_digraph(
    subsystem, digraph_filename="digraph.png", plot_digraph=False, layout="dot"
):

    G = nx.DiGraph()

    for n in range(subsystem.size):
        node_info = format_node(n, subsystem)
        G.add_node(
            node_info["label"],
            style=node_info["style"],
            fillcolor=node_info["fillcolor"],
            fontcolor=node_info["fontcolor"],
        )

    edges = [subsystem.indices2nodes(indices) for indices in np.argwhere(subsystem.cm)]

    G.add_edges_from(edges)
    G.graph["node"] = {"shape": "circle"}

    A = to_agraph(G)
    A.layout(layout)
    A.draw(digraph_filename)
    if plot_digraph:
        return Image(digraph_filename)


def plot_relations(
    subsystem,
    ces,
    relations,
    max_order=3,
    cause_effect_offset=(0.3, 0, 0),
    vertex_size_range=(10, 40),
    edge_size_range=(1, 5),
    surface_size_range=(0.05, 0.3),
    plot_dimentions=(1000, 1600),
    mechanism_labels_size=20,
    purview_labels_size=15,
    mesh_opacity=0.1,
    edge_width=1,
    show_mechanism_labels=True,
    show_purview_labels=True,
    show_vertices_mechanisms=True,
    show_vertices_purviews=True,
    show_edges="legendonly",
    show_mesh="legendonly",
    show_node_qfolds=False,
    show_mechanism_qfolds=True,
    showgrid=False,
    network_name="",
    eye_coordinates=(0.5, 0.5, 0.5),
    hovermode="x",
    digraph_filename="digraph.png",
    digraph_layout="dot",
):
    # Select only relations <= max_order
    relations = list(filter(lambda r: len(r.relata) <= max_order, relations))
    # Separate CES into causes and effects
    separated_ces = rel.separate_ces(ces)

    # Initialize figure
    fig = go.Figure()

    # Dimensionality reduction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create the features for each cause/effect based on their relations
    features = feature_matrix(separated_ces, relations)

    # Now we get one set of coordinates for the CES; these will then be offset to
    # get coordinates for causes and effects separately, so that causes/effects
    # are always near each other in the embedding.

    # Collapse rows of cause/effect belonging to the same distinction
    # NOTE: This depends on the implementation of `separate_ces`; causes and
    #       effects are assumed to be adjacent in the returned list
    umap_features = features[0::2] + features[1::2]
    distinction_coords = get_coords(umap_features)
    # Duplicate causes and effects so they can be plotted separately
    coords = np.empty(
        (distinction_coords.shape[0] * 2, distinction_coords.shape[1]),
        dtype=distinction_coords.dtype,
    )
    coords[0::2] = distinction_coords
    coords[1::2] = distinction_coords
    # Add a small offset to effects to separate them from causes
    coords[1::2] += cause_effect_offset

    # Purviews
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract vertex indices for plotly
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Get node labels and indices for future use:
    node_labels = subsystem.node_labels
    node_indices = subsystem.node_indices

    # Get mechanism and purview labels
    mechanism_labels = list(map(label_mechanism, ces))
    mechanism_labels_x2 = list(map(label_mechanism, separated_ces))
    purview_labels = list(map(label_purview, separated_ces))

    mechanism_hovertext = list(map(hovertext_mechanism, ces))
    vertices_hovertext = list(map(hovertext_purview, separated_ces))

    # Make mechanism labels
    xm, ym, zm = (
        [c + cause_effect_offset[0] / 2 for c in x[::2]],
        y[::2],
        [n + (vertex_size_range[1] / 10 ** 3) for n in z[::2]],
    )
    labels_mechanisms_trace = go.Scatter3d(
        visible=show_mechanism_labels,
        x=xm,
        y=ym,
        z=[n + (vertex_size_range[1] / 10 ** 3) for n in zm],
        mode="text",
        text=mechanism_labels,
        name="Mechanism Labels",
        showlegend=True,
        textfont=dict(size=mechanism_labels_size, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(labels_mechanisms_trace)

    # Compute purview and mechanism marker sizes
    purview_sizes = normalize_sizes(
        vertex_size_range[0], vertex_size_range[1], separated_ces
    )
    mechanism_sizes = [min(phis) for phis in chunk_list(purview_sizes, 2)]

    # Make mechanisms trace
    vertices_mechanisms_trace = go.Scatter3d(
        visible=show_vertices_mechanisms,
        x=xm,
        y=ym,
        z=zm,
        mode="markers",
        name="Mechanisms",
        text=mechanism_labels,
        showlegend=True,
        marker=dict(size=mechanism_sizes, color="black"),
        hoverinfo="text",
        hovertext=mechanism_hovertext,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(vertices_mechanisms_trace)

    # Make purview labels trace
    color = list(flatten([("red", "green")] * len(ces)))
    labels_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=x,
        y=y,
        z=[n + (vertex_size_range[1] / 10 ** 3) for n in z],
        mode="text",
        text=purview_labels,
        name="Purview Labels",
        showlegend=True,
        textfont=dict(size=purview_labels_size, color=color),
        hoverinfo="text",
        hovertext=vertices_hovertext,
        hoverlabel=dict(bgcolor=color),
    )
    fig.add_trace(labels_purviews_trace)

    # Make purviews trace
    purview_phis = [purview.phi for purview in separated_ces]
    direction_labels = list(flatten([["Cause", "Effect"] for c in ces]))
    vertices_purviews_trace = go.Scatter3d(
        visible=show_vertices_purviews,
        x=x,
        y=y,
        z=z,
        mode="markers",
        name="Purviews",
        text=purview_labels,
        showlegend=True,
        marker=dict(size=purview_sizes, color=color),
        hoverinfo="text",
        hovertext=vertices_hovertext,
        hoverlabel=dict(bgcolor=color),
    )
    fig.add_trace(vertices_purviews_trace)

    # 2-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get edges from all relations
    edges = list(
        flatten(
            relation_vertex_indices(features, j)
            for j in range(features.shape[1])
            if features[:, j].sum() == 2
        )
    )
    if edges:
        # Convert to DataFrame
        edges = pd.DataFrame(
            dict(
                x=x[edges],
                y=y[edges],
                z=z[edges],
                line_group=flatten(zip(range(len(edges) // 2), range(len(edges) // 2))),
            )
        )

        # Plot edges separately:
        two_relations = list(filter(lambda r: len(r.relata) == 2, relations))
        two_relations_sizes = normalize_sizes(
            edge_size_range[0], edge_size_range[1], two_relations
        )

        two_relations_coords = [
            list(chunk_list(list(edges["x"]), 2)),
            list(chunk_list(list(edges["y"]), 2)),
            list(chunk_list(list(edges["z"]), 2)),
        ]

        legend_nodes = []
        legend_mechanisms = []
        for r, relation in tqdm(
            enumerate(two_relations), desc="Computing edges", total=len(two_relations)
        ):
            relation_nodes = list(flatten(relation.mechanisms))

            # Make node contexts traces and legendgroups
            if show_node_qfolds:
                for node in node_indices:
                    node_label = make_label([node], node_labels)
                    if node in relation_nodes:

                        edge_2relation_trace = go.Scatter3d(
                            visible=show_edges,
                            legendgroup=f"Node {node_label} q-fold",
                            showlegend=True if node not in legend_nodes else False,
                            x=two_relations_coords[0][r],
                            y=two_relations_coords[1][r],
                            z=two_relations_coords[2][r],
                            mode="lines",
                            name=f"Node {node_label} q-fold",
                            line_width=two_relations_sizes[r],
                            line_color="blue",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        fig.add_trace(edge_2relation_trace)

                        if node not in legend_nodes:

                            legend_nodes.append(node)

            # Make nechanism contexts traces and legendgroups
            if show_mechanism_qfolds:
                mechanisms_list = [distinction.mechanism for distinction in ces]
                for mechanism in mechanisms_list:
                    mechanism_label = make_label(mechanism, node_labels)
                    if mechanism in relation.mechanisms:

                        edge_2relation_trace = go.Scatter3d(
                            visible=show_edges,
                            legendgroup=f"Mechanism {mechanism_label} q-fold",
                            showlegend=True
                            if mechanism_label not in legend_mechanisms
                            else False,
                            x=two_relations_coords[0][r],
                            y=two_relations_coords[1][r],
                            z=two_relations_coords[2][r],
                            mode="lines",
                            name=f"Mechanism {mechanism_label} q-fold",
                            line_width=two_relations_sizes[r],
                            line_color="blue",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        fig.add_trace(edge_2relation_trace)

                        if mechanism_label not in legend_mechanisms:

                            legend_mechanisms.append(mechanism_label)

            # Make all 2-relations traces and legendgroup
            edge_2relation_trace = go.Scatter3d(
                visible=show_edges,
                legendgroup="All 2-Relations",
                showlegend=True if r == 0 else False,
                x=two_relations_coords[0][r],
                y=two_relations_coords[1][r],
                z=two_relations_coords[2][r],
                mode="lines",
                # name=label_relation(relation),
                name="All 2-Relations",
                line_width=two_relations_sizes[r],
                line_color="blue",
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
                # text=label_two_relation(relation),
            )

            fig.add_trace(edge_2relation_trace)

    # 3-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get triangles from all relations
    triangles = [
        relation_vertex_indices(features, j)
        for j in range(features.shape[1])
        if features[:, j].sum() == 3
    ]

    if triangles:
        three_relations = list(filter(lambda r: len(r.relata) == 3, relations))
        three_relations_sizes = normalize_sizes(
            surface_size_range[0], surface_size_range[1], three_relations
        )
        # Extract triangle indices
        i, j, k = zip(*triangles)
        for r, triangle in tqdm(
            enumerate(triangles), desc="Computing triangles", total=len(triangles)
        ):
            relation = three_relations[r]
            relation_nodes = list(flatten(relation.mechanisms))

            if show_node_qfolds:
                for node in node_indices:
                    node_label = make_label([node], node_labels)
                    if node in relation_nodes:
                        triangle_3relation_trace = go.Mesh3d(
                            visible=show_mesh,
                            legendgroup=f"Node {node_label} q-fold",
                            showlegend=True if node not in legend_nodes else False,
                            # x, y, and z are the coordinates of vertices
                            x=x,
                            y=y,
                            z=z,
                            # i, j, and k are the vertices of triangles
                            i=[i[r]],
                            j=[j[r]],
                            k=[k[r]],
                            # Intensity of each vertex, which will be interpolated and color-coded
                            intensity=np.linspace(0, 1, len(x), endpoint=True),
                            opacity=three_relations_sizes[r],
                            colorscale="viridis",
                            showscale=False,
                            name=f"Node {node_label} q-fold",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        fig.add_trace(triangle_3relation_trace)

                        if node not in legend_nodes:

                            legend_nodes.append(node)

            if show_mechanism_qfolds:
                mechanisms_list = [distinction.mechanism for distinction in ces]
                for mechanism in mechanisms_list:
                    mechanism_label = make_label(mechanism, node_labels)
                    if mechanism in relation.mechanisms:
                        triangle_3relation_trace = go.Mesh3d(
                            visible=show_mesh,
                            legendgroup=f"Mechanism {mechanism_label} q-fold",
                            showlegend=True
                            if mechanism_label not in legend_mechanisms
                            else False,
                            # x, y, and z are the coordinates of vertices
                            x=x,
                            y=y,
                            z=z,
                            # i, j, and k are the vertices of triangles
                            i=[i[r]],
                            j=[j[r]],
                            k=[k[r]],
                            # Intensity of each vertex, which will be interpolated and color-coded
                            intensity=np.linspace(0, 1, len(x), endpoint=True),
                            opacity=three_relations_sizes[r],
                            colorscale="viridis",
                            showscale=False,
                            name=f"Mechanism {mechanism_label} q-fold",
                            hoverinfo="text",
                            hovertext=hovertext_relation(relation),
                        )
                        fig.add_trace(triangle_3relation_trace)
                        if mechanism_label not in legend_mechanisms:
                            legend_mechanisms.append(mechanism_label)

            triangle_3relation_trace = go.Mesh3d(
                visible=show_mesh,
                legendgroup="All 3-Relations",
                showlegend=True if r == 0 else False,
                # x, y, and z are the coordinates of vertices
                x=x,
                y=y,
                z=z,
                # i, j, and k are the vertices of triangles
                i=[i[r]],
                j=[j[r]],
                k=[k[r]],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=np.linspace(0, 1, len(x), endpoint=True),
                opacity=three_relations_sizes[r],
                colorscale="viridis",
                showscale=False,
                name="All 3-Relations",
                hoverinfo="text",
                hovertext=hovertext_relation(relation),
            )
            fig.add_trace(triangle_3relation_trace)

        # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes_range = [(min(d) - 1, max(d) + 1) for d in (x, y, z)]

    axes = [
        dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=showgrid,
            gridcolor="lightgray",
            showticklabels=False,
            showspikes=True,
            autorange=False,
            range=axes_range[dimension],
            backgroundcolor="white",
            title="",
        )
        for dimension in range(3)
    ]

    layout = go.Layout(
        showlegend=True,
        scene_xaxis=axes[0],
        scene_yaxis=axes[1],
        scene_zaxis=axes[2],
        scene_camera=dict(
            eye=dict(x=eye_coordinates[0], y=eye_coordinates[1], z=eye_coordinates[2])
        ),
        hovermode=hovermode,
        title=f"{network_name} Q-STRUCTURE",
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=False,
        height=plot_dimentions[0],
        width=plot_dimentions[1],
    )

    # Apply layout
    fig.layout = layout

    # Create system image
    save_digraph(subsystem, digraph_filename, layout=digraph_layout)
    digraph_coords = (-0.35, 1)
    digraph_size = (0.3, 0.4)

    fig.add_layout_image(
        dict(
            name="Causal model",
            source=digraph_filename,
            #         xref="paper", yref="paper",
            x=digraph_coords[0],
            y=digraph_coords[1],
            sizex=digraph_size[0],
            sizey=digraph_size[1],
            xanchor="left",
            yanchor="top",
        )
    )

    draft_template = go.layout.Template()
    draft_template.layout.annotations = [
        dict(
            name="Causal model",
            text="Causal model",
            opacity=1,
            font=dict(color="black", size=20),
            xref="paper",
            yref="paper",
            x=digraph_coords[0],
            y=digraph_coords[1] + 0.05,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
        )
    ]
    fig.update_layout(
        margin=dict(l=400),
        template=draft_template,
        annotations=[dict(templateitemname="Causal model", visible=True)],
    )

    return fig
