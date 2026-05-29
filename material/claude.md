
## Slide Requirements

When creating lecture slides (HTML format):
- **Beautiful minimalistic design** — clean, modern, academic aesthetic with University of Tubingen branding
- **HTML format** — using reveal.js framework with CDN imports
- **Interactive visualizations whenever possible** — prefer D3.js, Chart.js, or vanilla SVG/Canvas over static content
- **More visualizations, less text** — slides should be visual-first; minimize bullet points, maximize charts/diagrams/animations
- **Factual, based on established research papers** — always cite sources, use real data where available
- **Use original figures and data from papers** — whenever possible, reproduce or directly reference figures, tables, and experimental results from the source research papers rather than creating generic illustrations
- **Prefer downloading real figures over self-made charts** — for well-known plots (e.g., METR time horizons, Epoch AI training compute, Goodfellow's panda adversarial example), always download and embed the original figure from the source (blog post, paper, etc.) as a `<img>` tag rather than recreating it in Chart.js/D3. Self-made approximations are less accurate and look worse than the originals. Only create custom Chart.js/D3 visualizations when no good original figure exists or when interactivity is essential
- **Fast transitions** — use `transition: 'none'` or very fast transition speeds; no slow animations between slides
- **Double-check everything** — verify citations, math notation, factual claims, and that all interactive elements work
- **No em-dashes in slide text.** Em-dashes (—) read as LLM-generated. Use periods, commas, parentheses, colons, or sentence breaks instead. This applies to body text, captions, hints, and citations rendered to students. (Em-dashes are still fine inside HTML attributes and code comments where they are not displayed.)
- **Reduce information density.** Aim for slides that a student can read and absorb in 30–60 seconds. Trim verbose bullets, prefer one strong visual over a wall of text, and never repeat the same point across two cards on the same slide.

## Course Depth

This presentation targets a technically advanced audience. Content should build up from simple examples and should go beyond surface-level overviews and engage deeply with the underlying methods, math, and implementation details of each topic. Prefer formal problem statements, algorithmic descriptions, and concrete examples over high-level summaries. Please find the reference and description in README.md 

## Visualization Guidelines

- Visualizations should be **as interactive as possible** — let users manipulate parameters, step through algorithms, and explore data directly
- When illustrating dynamic processes (e.g., optimization trajectories, training dynamics, adversarial perturbation steps), prefer **animated GIFs or embedded animations** over static diagrams
- Interactive > animated GIF > static image — always choose the most engaging format the medium supports
- **Lecture-native integration** — every visualization and animation must flow naturally within its slide as an integral part of the lecture narrative. Avoid decorative or disconnected visuals; each one should directly support the point being made and feel like it belongs in the presentation flow
- **Interaction hints** — add concise captions beneath interactive visualizations indicating how to interact with them (e.g., "Click to step through iterations", "Hover over nodes to see weights", "Drag the slider to adjust ε")


## Session Length

Long-running sessions are expected and encouraged. It is normal for a session to last 45-60 mins. Do not prematurely wrap up or suggest splitting work across sessions — stay focused and drive the task to completion.

## File Reading

- **Always read lecture HTML files in their entirety** — never use partial reads (offset/limit) on lecture HTML files, even if they are 20K–100K tokens. Full context is essential for making correct edits and understanding slide structure.

## Session Context & Reuse

- **Check all markdown files in the repository** at the start of a session — not just claude.md. Other `.md` files may contain notes, plans, or instructions left by previous Claude Code sessions that are relevant to the current task.

## Research & Fact-Checking

- **Use web search liberally** — search the web as often as possible to verify claims, find exact numbers, confirm paper details, and ground all content in reality. Do not rely on memory alone for factual claims.
- **Fetch full papers, not just abstracts** — when retrieving research papers (via WebFetch or other tools), always fetch the full text. Abstracts alone are insufficient for accurate technical content.

## Compact Instructions

When compacting this conversation, follow these guidelines:
- **Preserve as many details as possible** — compaction should be thorough, not lossy. Err on the side of retaining too much rather than too little.
- **Target length: 20,000–40,000 tokens** — the compacted summary should be roughly in this range. Do not over-compress.
- **Always preserve:**
  - Current task progress and remaining steps
  - File paths being edited and their current state
  - Key design decisions made so far
  - Specific user instructions and preferences expressed during the session
  - Error messages, debugging context, and solutions found
  - Any TODO items or deferred work

## Preference Learning

After completing each user request, briefly analyze whether the interaction revealed a recurring user preference or convention that should be persisted. If so, propose adding it to this CLAUDE.md file so the user does not have to repeat the same instruction in future sessions.
