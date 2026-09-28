# Model definitions (results intentionally omitted)

This page fixes the meaning of the four model names used by the benchmark. It
contains no trained metrics, rankings or checkpoint claims.

## Dual-Graph GConvGRU

The full recurrent surrogate maintains one hidden state on the FE node graph and
one hidden state on an element graph. Node adjacency follows the 12 edges of each
C3D8 element. Element adjacency is face based: two solid elements communicate
only when they share a four-node face. The node branch predicts nodal
displacement. Node states are averaged over the eight incident nodes of each
element, passed to the element GConvGRU, and decoded as native element stress and
PEEQ. Global pooling of the element state feeds the reaction-force head.

The manuscript configuration uses 256-dimensional node and element hidden
states and two message-passing hops. These are recipe settings, not benchmark
results.

## Single-Graph GConvGRU

This is the matched ablation of the Dual-Graph model. It retains the node
GConvGRU, node inputs, hidden width, message-passing depth, normalization and
optimization protocol but removes the element graph and element recurrent
state. Element stress and PEEQ targets are averaged onto incident nodes for
training; predicted nodal proxies are averaged over each element's eight corner
nodes for native element-space evaluation.

Dual-Graph versus Single-Graph is the controlled architectural comparison.

## CGN-N

CGN-N is a node-only autoregressive encode-process-decode interaction network.
It is not a GConvGRU and therefore belongs to a different temporal/graph
architecture family. Displacement is predicted at nodes. Stress and PEEQ are
also decoded as nodal proxies, with element FE targets first averaged to nodes;
the loss is evaluated at node resolution. The CGN baseline family follows Li
et al. [1].

## CGN-E

CGN-E shares the same node-only CGN backbone as CGN-N. Its only intended
difference from CGN-N is supervision resolution: the predicted nodal stress and
PEEQ proxies are reconstructed to each C3D8 element during training, and the
loss is computed against the original element targets.

CGN-N versus CGN-E isolates supervision resolution within the CGN family.
Comparisons between either CGN and either GConvGRU model are cross-architecture
benchmarks, not matched ablations.

## Common task interface

Every model consumes the same case-level mesh and loading condition and rolls
over the same 21 normalized loading-progress frames. Every model is eventually
scored against native nodal displacement, native element stress/PEEQ, and the
global force history.

## References

[1] Q. Li, Z. Wang, L. Li, H. Hao, W. Chen, and Y. Shao, “Machine learning
prediction of structural dynamic responses using graph neural networks,”
*Computers & Structures*, vol. 289, Art. no. 107188, 2023.
