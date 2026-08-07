package org.bytedeco.pytorch.plot.vista;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * ASCII flow-diagram renderer for a {@link TraceGraph}.
 *
 * <p>Produces output like:
 * <pre>
 * user_id -> embedding.embed_user_id -┐
 * item_id -> embedding.embed_item_id -┤-> cat(embed) (2,24) -┬-> expert_0.layer_0 (2,24) -> expert_0.layer_1 -> ...
 * cate_id -> embedding.embed_cate_id -┘                     ├-> expert_1.layer_0 (2,24) -> ...
 *                                                            └-> gate (2,24) -> ...
 * </pre>
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li>Identify source nodes (INPUT type, or zero-indegree nodes).</li>
 *   <li>Walk forward from each source along linear chains (single-out nodes).
 *       A chain stops at a <em>fan-in point</em> (node with &gt;1 predecessor)
 *       or a <em>fan-out point</em> (node with &gt;1 successor).</li>
 *   <li>Fan-in points are rendered as a group: each predecessor's chain ends
 *       with {@code -┐ / -┤ / -┘}, and the middle line continues with
 *       {@code -> fanInNode}.</li>
 *   <li>Fan-out points are rendered with {@code -┬-> / ├-> / └->} brackets,
 *       one child chain per line, indented to align.</li>
 *   <li>Visited nodes are tracked so shared subgraphs expand only once; a
 *       back-reference {@code (see above)} is emitted for repeats.</li>
 * </ol>
 *
 * <p>The renderer is read-only: it never mutates the graph.
 */
final class FlowRenderer {
    private static final String CHAIN = " -> ";
    private static final String FAN_IN_TOP = " -\u250C";   //  -┐
    private static final String FAN_IN_MID = " -\u2524";   //  -┤
    private static final String FAN_IN_BOT = " -\u2518";   //  -┘
    private static final String FAN_OUT_MID = "\u251C-> ";  // ├->
    private static final String FAN_OUT_BOT = "\u2514-> ";  // └->

    private final TraceGraph graph;
    private final Map<String, GraphNode> adj;
    private final int maxWidth;

    /** Nodes that have already been rendered somewhere in the output. */
    private final Set<String> rendered = new HashSet<>();
    /** Reverse adjacency: target -> sorted live sources. */
    private final Map<String, List<String>> reverse = new HashMap<>();
    /** Forward adjacency: source -> sorted live targets. */
    private final Map<String, List<String>> forward = new HashMap<>();

    FlowRenderer(TraceGraph graph, int maxWidth) {
        this.graph = graph;
        this.adj = graph.adjList();
        this.maxWidth = Math.max(40, maxWidth);
        buildAdjacency();
    }

    private void buildAdjacency() {
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            String src = e.getKey();
            List<String> kids = new ArrayList<>();
            for (GraphEdge edge : e.getValue().edges()) {
                if (adj.containsKey(edge.target()) && !kids.contains(edge.target())) {
                    kids.add(edge.target());
                }
            }
            Collections.sort(kids);
            forward.put(src, kids);
            for (String tgt : kids) {
                reverse.computeIfAbsent(tgt, k -> new ArrayList<>()).add(src);
            }
        }
    }

    String render() {
        List<String> sources = collectSources();
        if (sources.isEmpty()) {
            return renderFlat();
        }
        StringBuilder out = new StringBuilder();
        // Render each source's subtree. Fan-in points are detected during the
        // walk and rendered as a unit (all predecessors + the merge node).
        Set<String> fanInRootsRendered = new HashSet<>();
        for (String src : sources) {
            // Skip sources that were already rendered as part of a fan-in
            // group (e.g. the 2nd/3rd inputs feeding a cat node).
            if (rendered.contains(src)) continue;
            // If this source feeds into a fan-in point that has already been
            // rendered as part of another source's group, skip it.
            String merge = firstFanInDownstream(src);
            if (merge != null && fanInRootsRendered.contains(merge)) {
                continue;
            }
            if (out.length() > 0) out.append('\n');
            if (merge != null) {
                // Render the whole fan-in group: all predecessors of `merge`
                // that are reachable as linear chains from sources.
                renderFanInGroup(out, merge);
                fanInRootsRendered.add(merge);
            } else {
                renderFrom(out, src, 0);
            }
        }
        // Render any remaining nodes that weren't reached.
        List<String> leftover = new ArrayList<>();
        for (String n : adj.keySet()) {
            if (!rendered.contains(n)) leftover.add(n);
        }
        Collections.sort(leftover);
        for (String n : leftover) {
            if (out.length() > 0) out.append('\n');
            renderFrom(out, n, 0);
        }
        return out.toString();
    }

    /** Collect source nodes: INPUT-type first, then any zero-indegree node. */
    private List<String> collectSources() {
        List<String> inputs = new ArrayList<>();
        List<String> zeroIn = new ArrayList<>();
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            String name = e.getKey();
            GraphNode node = e.getValue();
            boolean hasIncoming = reverse.containsKey(name) && !reverse.get(name).isEmpty();
            if (node.nodeType() == NodeType.INPUT) {
                inputs.add(name);
            } else if (!hasIncoming) {
                zeroIn.add(name);
            }
        }
        Collections.sort(inputs);
        Collections.sort(zeroIn);
        List<String> all = new ArrayList<>(inputs.size() + zeroIn.size());
        all.addAll(inputs);
        all.addAll(zeroIn);
        return all;
    }

    /**
     * Walk the linear chain from {@code src} and return the first node on that
     * chain that has &gt;1 predecessor (a fan-in / merge point), or null if the
     * chain never reaches one.
     */
    private String firstFanInDownstream(String src) {
        String cur = src;
        int guard = 0;
        while (cur != null && guard++ < 256) {
            List<String> preds = reverse.getOrDefault(cur, Collections.emptyList());
            if (preds.size() > 1 && !cur.equals(src)) {
                return cur;
            }
            List<String> kids = forward.getOrDefault(cur, Collections.emptyList());
            if (kids.size() != 1) break;
            cur = kids.get(0);
        }
        return null;
    }

    /**
     * Render a fan-in group: all source-predecessors of {@code merge} that are
     * reachable via linear chains, joined with {@code -┐ -┤ -┘}, then the
     * merge node and its continuation on the middle line. The merge node's
     * fan-out (if any) is rendered <em>after</em> all fan-in lines so the
     * closing bracket ({@code -┘}) stays adjacent to its siblings.
     */
    private void renderFanInGroup(StringBuilder out, String merge) {
        // Find all predecessors of `merge` that are sources or reachable from
        // sources via linear chains. Each predecessor's chain is rendered up
        // to (but not including) the merge node.
        List<String> preds = reverse.getOrDefault(merge, Collections.emptyList());
        List<String> sortedPreds = new ArrayList<>(preds);
        Collections.sort(sortedPreds);
        // For each predecessor, build the linear chain from its source root
        // down to the predecessor itself.
        List<String> chains = new ArrayList<>(sortedPreds.size());
        int maxLen = 0;
        for (String pred : sortedPreds) {
            String root = findSourceRoot(pred);
            StringBuilder cb = new StringBuilder();
            collectChainFrom(cb, root, merge);
            String chain = cb.toString();
            chains.add(chain);
            maxLen = Math.max(maxLen, chain.length());
        }
        if (chains.isEmpty()) {
            // No predecessor chains — just render the merge node directly.
            renderFrom(out, merge, 0);
            return;
        }
        int mid = chains.size() / 2;
        // First pass: emit all fan-in lines. The middle line includes the
        // merge node label and any single-child linear continuation, but NOT
        // the fan-out (that comes after the closing bracket).
        for (int i = 0; i < chains.size(); i++) {
            String chain = chains.get(i);
            String bracket = (i == 0) ? FAN_IN_TOP
                    : (i == chains.size() - 1) ? FAN_IN_BOT : FAN_IN_MID;
            out.append(chain).append(bracket);
            if (i == mid) {
                // Append the merge node + single-child continuation on the
                // middle line. Fan-out is deferred to the second pass.
                out.append("-> ").append(nodeLabel(merge));
                rendered.add(merge);
                List<String> kids = forward.getOrDefault(merge, Collections.emptyList());
                if (kids.size() == 1) {
                    out.append(CHAIN).append(renderLinearChain(kids.get(0)));
                }
            }
            if (i < chains.size() - 1) out.append('\n');
        }
        // Second pass: render the merge node's fan-out (if any) below the
        // fan-in group, indented to align under the merge node.
        List<String> mergeKids = forward.getOrDefault(merge, Collections.emptyList());
        if (mergeKids.size() > 1) {
            renderFanOut(out, merge, maxLen + 4);
        } else if (mergeKids.size() == 1) {
            // Single child already rendered on the middle line. Walk forward
            // from the child to find the next fan-out or fan-in point and
            // render it recursively.
            renderDownstream(out, mergeKids.get(0), maxLen + 4);
        }
    }

    /**
     * Walk forward from {@code node} (already rendered) along linear chains
     * and render the first fan-out point or fan-in point encountered. Used to
     * continue rendering after a linear chain ends without a fan-out. If the
     * downstream fan-in point has already been rendered, stop immediately
     * (its fan-out was already emitted elsewhere).
     */
    private void renderDownstream(StringBuilder out, String node, int indent) {
        String cur = node;
        int guard = 0;
        while (cur != null && guard++ < 256) {
            List<String> kids = forward.getOrDefault(cur, Collections.emptyList());
            if (kids.size() > 1) {
                // Fan-out point — render children.
                renderFanOut(out, cur, indent);
                return;
            }
            if (kids.size() == 1) {
                String next = kids.get(0);
                List<String> nextPreds = reverse.getOrDefault(next, Collections.emptyList());
                if (nextPreds.size() > 1) {
                    // Fan-in point downstream.
                    if (rendered.contains(next)) {
                        // Already rendered — stop to avoid duplicate fan-out.
                        return;
                    }
                    out.append(CHAIN).append(nodeLabel(next));
                    rendered.add(next);
                    List<String> fanInKids = forward.getOrDefault(next, Collections.emptyList());
                    if (fanInKids.size() == 1) {
                        out.append(CHAIN).append(renderLinearChain(fanInKids.get(0)));
                        renderDownstream(out, fanInKids.get(0), indent);
                    } else if (fanInKids.size() > 1) {
                        renderFanOut(out, next, indent);
                    }
                    return;
                }
                cur = next;
            } else {
                return;
            }
        }
    }

    /**
     * Find the source root of {@code node}: walk backwards along single-in
     * chains until reaching a node with no predecessor (or a fan-in point).
     */
    private String findSourceRoot(String node) {
        String cur = node;
        int guard = 0;
        while (guard++ < 256) {
            List<String> preds = reverse.getOrDefault(cur, Collections.emptyList());
            if (preds.isEmpty()) break;
            // Only follow single-predecessor chains; if multiple, this node
            // is itself a merge point and we stop here.
            if (preds.size() > 1) break;
            cur = preds.get(0);
        }
        return cur;
    }

    /**
     * Collect the linear chain starting at {@code start}, stopping before
     * {@code stop}. Appends formatted labels joined by {@code ->}.
     */
    private void collectChainFrom(StringBuilder out, String start, String stop) {
        String cur = start;
        boolean first = true;
        int guard = 0;
        while (cur != null && guard++ < 256) {
            if (cur.equals(stop)) break;
            if (!first) out.append(CHAIN);
            out.append(nodeLabel(cur));
            rendered.add(cur);
            first = false;
            List<String> kids = forward.getOrDefault(cur, Collections.emptyList());
            if (kids.size() != 1) break;
            String next = kids.get(0);
            // Stop if next is a fan-in point (unless it's the stop target).
            List<String> nextPreds = reverse.getOrDefault(next, Collections.emptyList());
            if (nextPreds.size() > 1 && !next.equals(stop)) break;
            cur = next;
        }
    }

    /**
     * Render a linear chain (single-in, single-out) starting at {@code start}
     * as a single line: {@code label0 -> label1 -> ...}. Stops at fan-out or
     * fan-in points. Returns the rendered string. The chain stops <em>before</em>
     * any fan-in point (node with &gt;1 predecessor) — the caller is responsible
     * for rendering the fan-in group.
     */
    private String renderLinearChain(String start) {
        StringBuilder sb = new StringBuilder();
        String cur = start;
        boolean first = true;
        int guard = 0;
        while (cur != null && guard++ < 256) {
            if (!first) sb.append(CHAIN);
            sb.append(nodeLabel(cur));
            rendered.add(cur);
            first = false;
            List<String> kids = forward.getOrDefault(cur, Collections.emptyList());
            if (kids.size() != 1) break;
            String next = kids.get(0);
            List<String> nextPreds = reverse.getOrDefault(next, Collections.emptyList());
            if (nextPreds.size() > 1) break;
            cur = next;
        }
        return sb.toString();
    }

    /**
     * Return the single downstream child of the last node in the chain started
     * by {@code start}, if that child is a fan-in point (merge node). Used to
     * detect when a linear chain leads into a merge that needs its own group.
     */
    private String chainEndsAtFanIn(String start) {
        String cur = start;
        int guard = 0;
        while (cur != null && guard++ < 256) {
            List<String> kids = forward.getOrDefault(cur, Collections.emptyList());
            if (kids.size() != 1) return null;
            String next = kids.get(0);
            List<String> nextPreds = reverse.getOrDefault(next, Collections.emptyList());
            if (nextPreds.size() > 1) return next;
            cur = next;
        }
        return null;
    }

    /**
     * Render starting at {@code node} with the given indent. Handles linear
     * chains, fan-out points, and fan-in points downstream of a chain.
     */
    private void renderFrom(StringBuilder out, String node, int indent) {
        if (rendered.contains(node)) {
            for (int s = 0; s < indent; s++) out.append(' ');
            out.append(nodeLabel(node)).append(" (see above)");
            return;
        }
        for (int s = 0; s < indent; s++) out.append(' ');
        out.append(renderLinearChain(node));
        // After the linear chain, check for a downstream fan-in point.
        String fanIn = chainEndsAtFanIn(node);
        if (fanIn != null && !rendered.contains(fanIn)) {
            out.append(CHAIN).append(nodeLabel(fanIn));
            rendered.add(fanIn);
            List<String> kids = forward.getOrDefault(fanIn, Collections.emptyList());
            if (kids.size() == 1) {
                out.append(CHAIN).append(renderLinearChain(kids.get(0)));
                renderDownstream(out, kids.get(0), indent);
            } else if (kids.size() > 1) {
                renderFanOut(out, fanIn, indent + 4);
            }
            return;
        }
        // No downstream fan-in — check for a downstream fan-out point.
        renderDownstream(out, node, indent);
    }

    /**
     * Render the fan-out children of {@code parent}. Each child is rendered on
     * its own line with {@code ├->} / {@code └->} brackets, indented to align
     * under the parent's column. If a child's chain leads to a downstream
     * fan-in point, that group is rendered recursively.
     */
    private void renderFanOut(StringBuilder out, String parent, int indent) {
        List<String> kids = forward.getOrDefault(parent, Collections.emptyList());
        if (kids.isEmpty()) return;
        for (int i = 0; i < kids.size(); i++) {
            String child = kids.get(i);
            out.append('\n');
            for (int s = 0; s < indent; s++) out.append(' ');
            out.append(i == kids.size() - 1 ? FAN_OUT_BOT : FAN_OUT_MID);
            if (rendered.contains(child)) {
                out.append(nodeLabel(child)).append(" (see above)");
                continue;
            }
            out.append(renderLinearChain(child));
            // Check if this child's chain leads to a downstream fan-in point.
            String fanIn = chainEndsAtFanIn(child);
            if (fanIn != null && !rendered.contains(fanIn)) {
                out.append(CHAIN).append(nodeLabel(fanIn));
                rendered.add(fanIn);
                List<String> fanInKids = forward.getOrDefault(fanIn, Collections.emptyList());
                if (fanInKids.size() == 1) {
                    out.append(CHAIN).append(renderLinearChain(fanInKids.get(0)));
                    renderDownstream(out, fanInKids.get(0), indent + 4);
                } else if (fanInKids.size() > 1) {
                    renderFanOut(out, fanIn, indent + 4);
                }
                continue;
            }
            // No unrendered downstream fan-in — check for a downstream fan-out
            // point along the child's chain. renderDownstream stops safely if
            // it hits an already-rendered fan-in.
            renderDownstream(out, child, indent + 4);
        }
    }

    /** Format a node's display label: {@code display_name (attr_name) (dims)}. */
    private String nodeLabel(String name) {
        String display = graph.graphNodeDisplayNames().getOrDefault(name, name);
        String attr = graph.nodeToAttrName().get(name);
        StringBuilder sb = new StringBuilder();
        sb.append(display == null || display.isEmpty() ? name : display);
        // Append attr name in parens if it differs from the display name and
        // is non-trivial (e.g. "expert_0" for a LinearImpl).
        if (attr != null && !attr.isEmpty() && !attr.equals(display)
                && !attr.equals(name)) {
            sb.append(" (").append(attr).append(')');
        }
        // Append dims if available on any incoming edge.
        String dims = firstIncomingDims(name);
        if (dims != null && !dims.isEmpty()) {
            sb.append(" (").append(dims).append(')');
        }
        return sb.toString();
    }

    /** Find the first non-empty dims string among edges pointing at {@code name}. */
    private String firstIncomingDims(String name) {
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            for (GraphEdge edge : e.getValue().edges()) {
                if (name.equals(edge.target())) {
                    String d = edge.dims();
                    if (d != null && !d.isEmpty()) return d;
                }
            }
        }
        return null;
    }

    /** Fallback: flat one-line-per-node listing when no sources are found. */
    private String renderFlat() {
        StringBuilder sb = new StringBuilder();
        List<String> names = new ArrayList<>(adj.keySet());
        Collections.sort(names);
        for (String n : names) {
            if (sb.length() > 0) sb.append('\n');
            sb.append(nodeLabel(n));
            List<String> kids = forward.getOrDefault(n, Collections.emptyList());
            if (!kids.isEmpty()) {
                sb.append(" -> ").append(String.join(", ", kids));
            }
        }
        return sb.toString();
    }
}