### IMPORTS AND MACROS #########################################################

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

ss = st.session_state


# Calculate weight of two interest vectors
def calc_prog_weights(v1: list, v2: list) -> float:
    n = len(v1)
    d = sum([max(v1[i], v2[i]) for i in range(n)])
    return d / (5 * n)

def merge_values(u: int, v: int, df) -> None:
    df["preferred_team_size"][u] = min(
        df["preferred_team_size"][v], df["preferred_team_size"][u]
    )
    df["year_of_study"][u] = min(df["year_of_study"][u], df["year_of_study"][v])
    df["padre"][v] = u
    l = len(df["programming_skills"][u])
    df["programming_skills"][u] = [
        max(df["programming_skills"][u][i], df["programming_skills"][v][i])
        for i in range(l)
    ]
    if df["experience_level"][v] == "Advanced":
        df["experience_level"][u] = "Advanced"
    elif (
        df["experience_level"][v] != "Beginner"
        and df["experience_level"][u] != "Advanced"
    ):
        df["experience_level"][u] = df["experience_level"][v]
    df["hackathons_done"][u] += df["hackathons_done"][v]
    for i in df["interests"][v]:
        if i not in df["interests"][u]:
            df["interests"][u].append(i)
    for i in df["interests_challenges"][v]:
        if i not in df["interests_challenges"][u]:
            df["interests_challenges"][u].append(i)
    for i in df["preferred_role"][v]:
        if i not in df["preferred_role"][u]:
            df["preferred_role"][u].append(i)
    df["objective"][u] = [
        df["freq"][u] * df["objective"][u][i] + df["freq"][v] * df["objective"][v][i]
        for i in range(4)
    ]
    df["objective"][u] = df["objective"][u] / np.linalg.norm(df["objective"][u])
    df["freq"][u] += df["freq"][v]
    df["preferred_languages"][u] = [
        p for p in df["preferred_languages"][u] if p in df["preferred_languages"][v]
    ]
    for a, b in df["availability"][u].items():
        if b and not (df["availability"][v][a]):
            df["availability"][u][a] = False
    for i in df["friends"][v]:
        df["friends"][i] = [w for w in df["friends"][i] if w != v]
        if u not in df["friends"][i]:
            df["friends"][i].append(u)
        if i not in df["friends"][u]:
            df["friends"][u].append(i)


def recalc_weight(u: int, v: int, df, multipliers: dict[str, float]) -> float:
    # year_of_study
    year_weight = (5 - abs(df["year_of_study"][u] - df["year_of_study"][v])) / 5

    # year_of_study
    year_weight = (5 - abs(df["year_of_study"][u] - df["year_of_study"][v])) / 5

    # interests
    interests_weight = 0
    for a in df["interests"][u]:
        for b in df["interests"][v]:
            if a == b: interests_weight += 1
    interests_weight / max(len(df["interests"][u]), len(df["interests"][v]), 1)

    # preferred_role
    role_weight = 1 if df["preferred_role"][u] != df["preferred_role"][v] else 0

    # friend_registration
    friend_weight = 0
    if u in df["friends"][v]:
        friend_weight += 1
    if v in df["friends"][u]:
        friend_weight += 1

    # interest_in_challenges
    challenges_weight = 0.0
    for a in df["interests_challenges"][u]:
        for b in df["interests_challenges"][v]:
            if a == b: challenges_weight += 1
    challenges_weight /= 3

    # preferred_languages
    languages_weight = 0
    for a in df["preferred_languages"][u]:
        for b in df["preferred_languages"][u]:
            if a == b: languages_weight = max(languages_weight, 1)

    # objective
    objective_weight = np.dot(
        np.array(df["objective"][u]), np.array(df["objective"][v])
    )

    # availability
    availability_weight = 0
    for a, b in df["availability"][u].items():
        if b and df["availability"][v][a]:
            availability_weight += 1
    availability_weight /= 5

    # programming_skills
    programming_weight = calc_prog_weights(
        df["programming_skills"][u], df["programming_skills"][v]
    )

    total_weight = (
        multipliers["year_mult"] * year_weight
        + multipliers["interests_mult"] * interests_weight
        + multipliers["role_mult"] * role_weight
        + multipliers["friend_mult"] * friend_weight
        + multipliers["challenges_mult"] * challenges_weight
        + multipliers["languages_mult"] * languages_weight
        + multipliers["objective_mult"] * objective_weight
        + multipliers["availability_mult"] * availability_weight
        + multipliers["programming_mult"] * programming_weight
    )

    return total_weight


def merge_nodes(G: nx.Graph, u: int, v: int, df, multipliers: dict[str, float]) -> tuple[nx.Graph, pd.DataFrame]:
    # el graf que retornarem es G
    # wlog eliminem v
    merge_values(u, v, df)
    for y in G[u]:
        G[u][y]["weight"] = G[y][u]["weight"] = recalc_weight(u, y, df, multipliers)

    G.remove_node(v)
    return (G, df)


def add_options(
    G: nx.Graph, B: int, df: pd.DataFrame, multipliers: dict[str, float]
) -> list[tuple[nx.Graph, pd.DataFrame]]:
    total_edges = G.edges.data()
    list_edges = []
    for u, v, d in total_edges:
        w = d["weight"]
        if df["freq"][u] + df["freq"][v] <= min(
            df["preferred_team_size"][u], df["preferred_team_size"][v]
        ):
            list_edges.append((w, u, v))
    list_edges = sorted(list_edges, key=lambda x: x[0])
    ans: list[tuple[int, nx.Graph, pd.DataFrame]] = []
    for i in range(min(B, len(list_edges))):
        a, b = merge_nodes(G.copy(), list_edges[i][1], list_edges[i][2], df.copy(), multipliers)
        ans.append((list_edges[i][0], a, b))
    return ans


def root(u: int, df: pd.DataFrame) -> int:
    if df["padre"][u] == u:
        return u
    df["padre"][u] = root(df["padre"][u], df)
    return df["padre"][u]


def solve(G: nx.Graph, W: int, B: int, df: pd.DataFrame, multipliers: dict[str, float]) -> None:
    llista = [(0, G, df)]
    n = G.number_of_nodes()

    for _ in range(n - 1):
        novallista = []
        for w, g, d in llista:
            semillista = add_options(g, B, d, multipliers)
            for weight, graf, frame in semillista:
                novallista.append((w + weight, graf, frame))
        novallista = sorted(novallista, key=lambda x: x[0])
        if len(novallista) == 0:
            break
        llista = novallista[:W]
        # print(llista)
    totalw, finalg, finaldf = llista[0]
    comps = []
    for i in range(n):
        comps.append(root(i, finaldf))
    ordered_components = {}
    cnt = 0
    for i in range(n):
        if comps[i] not in ordered_components:
            ordered_components[comps[i]] = cnt
            cnt += 1
        comps[i] = ordered_components[comps[i]]
    # print(comps)
    return comps


def create_graph(participants, multipliers: dict, edge_threshold: float, df: pd.DataFrame) -> nx.Graph:
    nodes = pd.DataFrame(
        data=[[i, participants[i].name] for i in range(len(participants))],
        columns=["id", "label"],
    )

    edges = pd.DataFrame(data=[], columns=["id1", "id2", "weight"])
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            edges.loc[len(edges)] = [i, j, recalc_weight(i, j, df, multipliers)]
    # G = make_graph(role_mult, interests_mult, year_mult, friend_mult, challenges_mult, languages_mult, objective_mult, availability_mult, edge_max_width, edge_threshold, n_clusters=8)
    max_weight = max(edges["weight"])
    if max_weight != 0.0:
        edges["weight"] = edges["weight"] / max_weight
    edges = edges[edges["weight"] > edge_threshold]
    edges_norm = edges.copy()
    edges["weight"] = 1 / edges["weight"]

    G = nx.Graph()
    G.add_nodes_from(nodes["id"])
    G.add_edges_from(
        zip(edges["id1"], edges["id2"], edges[["weight"]].to_dict(orient="records"))
    )
    return G, edges_norm


def beam_search(
    role_mult: float,
    interests_mult: float,
    year_mult: float,
    friend_mult: float,
    challenges_mult: float,
    languages_mult: float,
    objective_mult: float,
    availability_mult: float,
    programming_mult: float,
    edge_threshold: float,
    num_nodes: int,
) -> None:

    participants = ss.participants[0:num_nodes]
    mydict = {participants[i].id: i for i in range(len(participants))}

    multipliers = {
        'role_mult' : role_mult,
        'interests_mult' : interests_mult,
        'year_mult' : year_mult,
        'friend_mult' : friend_mult,
        'challenges_mult' : challenges_mult,
        'languages_mult' : languages_mult,
        'objective_mult' : objective_mult,
        'availability_mult' : availability_mult,
        "programming_mult" : programming_mult
    }

    df = pd.DataFrame(
        data=[
            [
                1,
                p.id,
                p.preferred_team_size,
                p.year_of_study,
                0,
                p.interest_in_challenges,
                p.interests,
                [p.preferred_role],
                p.experience_level,
                p.hackathons_done,
                p.objective,
                p.preferred_languages,
                [mydict[f] for f in p.friend_registration if f in mydict],
                p.availability,
                p.programming_skills,
            ]
            for p in participants
        ],
        columns=[
            "freq",
            "id",
            "preferred_team_size",
            "year_of_study",
            "padre",
            "interests_challenges",
            "interests",
            "preferred_role",
            "experience_level",
            "hackathons_done",
            "objective",
            "preferred_languages",
            "friends",
            "availability",
            "programming_skills",
        ],
    )
    for i in range(num_nodes):
        df["padre"][i] = i
    G, edges = create_graph(participants, multipliers, edge_threshold, df)
    components = solve(G, 5, 5, df, multipliers)
    tot_nodes = pd.DataFrame(
        data=[[i, participants[i].name, components[i]] for i in range(num_nodes)],
        columns=["id", "label", "cluster"],
    )

    return components, tot_nodes, edges


if __name__ == "__main__":
    beam_search()
