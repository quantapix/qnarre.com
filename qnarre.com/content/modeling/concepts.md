---
layout: modeling
title: Conceptual Model Of An "Emotional" Domain
description: 
toc: true
---

We are only concerned with textual data stored electronically. A `source` is thus our generic unit of representation throughout the problem domain.

The `evidence` is the exhaustive collection of all the `doc`s we need to consider, organized hierarchically as individual `source`s.

## Claim

Ultimately, it all comes down to working with text fragments, or `claims`, with their various interconnected and even overlapping properties and attributes.

We formalize these and define:

| Term         | Definition                         | Properties                               |
| ---          | ---                                | ---                                      |
| **Source**   | digitized content stored in a file | type (e.g. pdf, jpg, email, text)        |
| **Evidence** | a hierarchy of considered Sources  | root (directory)                         |
| **Doc**      | paginated, flat textual Evidence   | name (date), genre, author, title, pages |
| **Genre**    | the "official-ness" of a Doc        | name                                     |
| **Place**    | a specific location in a Doc       | doc, page, para(graph)                   |
| **Claim**    | a Placed relevant text fragment    | text, place, credibility                 |

## Author

The `author` of a `doc` of implicitly related set of `claim`s can be a party, a lawyer or a judge, for example.

Accordingly, we define:

| Term          | Definition                            | Properties                           |
| ---           | ---                                   | ---                                  |
| **Author**    | an author of Docs of Claims           | name, title                          |
| **Agent**     | a to be believed professional         | kind (of profession)                 |
| **Authority** | a trusted Agent of an official agency | agency (institution or 'sworn self') |

## Narrative

A `claim` can be grouped into `topic`s and attached to `narrative`s. Such groupings and assignments are done based on the semantic meaning of the claim, as well as of the context the claim belongs to.

| Term          | Definition                                    | Properties  |
| ---           | ---                                           | ---         |
| **Topic**     | a relevant/dominant subject matter of Claims  | name        |
| **Narrative** | a thematically focused "slice" through Claims | name, topic |

More importantly, `claim`s are further categorized as follows.

## Reality

The objective is to either compute a generalizing `credibility` for the claim or to derive other reproducible conjectures, regarding the related claims.

| Term           | Definition                                                | Properties       |
| ---            | ---                                                       | ---              |
| **Node**       | a generic grouping of Claim(s) along thematic dimensions  | topic, narrative |
| **Proof**      | a readily verifiable, undisputed Claim from an Authority  | claim, authority |
| **Conjecture** | a logical inference from Proofs guided by thematic Claims | proofs, claims   |
| **Reality**    | a broad-based, reinforcing Conjecture                     | coherence        |
| **Dissent**    | a narrow, fragmenting and thus weakening Conjecture       | fragmentation    |

While a broad `reality` is "simple", as in hard to characterize any further, a narrow `dissent` is much richer in "character".

## Dissent

We break down `dissent`s into self-explanatory types, with increasing factors of their fragmenting effects. These are:

| Term                | Definition                                                       |
| ---                 | ---                                                              |
| **Misreading**      | seemingly unintentionally misinterpreted relevant claims         |
| **Fairness**        | neglected or refused to consider both claim and counterclaim     |
| **Negligence**      | reported relevant and damaging information in a careless manner  |
| **Proportionality** | neglected or refused to report in balanced/proportional manner   |
| **Contradiction**   | neglected or refused to state apparent inconsistency or conflict |
| **Isolation**       | neglected or refused to consider relevant and essential context  |
| **Omission**        | neglected or refused to report disclosed relevant information    |
| **Distortion**      | reported relevant information in a suggestive and biased manner  |

A special type of claim is a `conflict`.

## Conflict

Its purpose is to directly contradict a `reality` and thus to manipulate a subsequent `judgment`. As the most important objects in modeling a lawsuit, `conflict`s are at the center of our attention.

The ultimate goal of our modeling efforts is to arrive at a mechanism to extract a semantic "conflict graph" in a reproducible fashion.

| Term         | Definition                                                 | Properties           |
| ---          | ---                                                        | ---                  |
| **Conflict** | a Claim intended to fabricate/escalate bases for Judgments | claim, reality       |
| **Suborn**   | a false Claim caused by Agent as intentional perjury       | (agent's) agency     |
| **Repeat**   | variant of a conflict aimed at reinforcing intended effect | claim, conflict      |
| **Judgment** | a thematic conjecture by Authority over inflated Conflicts | conflicts, authority |
| **Activism** | a thematic pattern of conjectures by Authority             | judgments, authority |

Once again, revealing, and even actively fabricating, conflicts is the objective of lawsuits.

We categorize conflicts along their increasing fragmenting effect on realities:

| Term         | Definition                                                       |
| ---          | ---                                                              |
| **Inherent** | emphasized inherent contradictions to confuse judgment           |
| **Conceal**  | intentionally concealed relevant truths to affect judgment       |
| **Deceive**  | intentionally emphasized relevant half-truths to affect judgment |
| **Fraud**    | stated intentionally falsified relevant and crucial information  |
| **Extort**   | stated relevant and threatening claims to force money or "fees"  |

## Judgment

The judgments of lawsuits represent the net effect of all claimed conflicts.

If a `judgment` relies predominantly on perhaps fabricated conflicts, or is simply prejudiced by them, while actively overruling realities, it causes `turmoil`.

The types of judgments we consider in increasing `turmoil` effect is as follows:

| Term            | Definition                                                         |
| ---             | ---                                                                |
| **Validation**  | selection and judgment of conflicts refuting an inflated Conflict  |
| **Confusion**   | selection and judgment of conflicts leading to clear contradiction |
| **Bias**        | selection and judgment of conflicts satisfying a prior narrative   |
| **Disregard**   | avoidance of relevant conflicts while protecting a prior narrative |
| **Fabrication** | injection of disputable conjecture to reinforce a prior narrative  |

When considering multiple judgment claims, a pattern, or a trend, can possibly be established.

## Activism

During processing, when our calculations support such "judicial" patterns, `activism`s emerge.

We define the following `activism` categories in increasing effect of their `turmoil` factor:

| Term           | Definition                                            |
| ---            | ---                                                   |
| **Exclude**    | a pattern of excluding undesired realities            |
| **Insinuate**  | a pattern of insinuating desired "realities"          |
| **Polarize**   | a pattern of deepening fabricated conflicts           |
| **Recast**     | a pattern of re-interpreting realities                |
| **Elevate**    | a pattern of elevating desired "realities"            |
| **Victimize**  | a pattern of repeatedly causing suffering             |
| **Exploit**    | a pattern of repeatedly preying on the defenseless    |
| **Perpetuate** | a pattern of reinforcing the repetition of unfairness |

This blog introduced the most essential "framing" concepts of our model. A subsequent blog will define the precise analytical, or quantitatively measurable, properties of these elements.
