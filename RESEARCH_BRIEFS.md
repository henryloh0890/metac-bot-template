# Research briefs for human contests
Bots cannot win prizes in human contests such as the Metaculus Cup, but Metaculus rules allow people to use AI for research, as long as the final forecast is your own decision. So here the bot only researches and gives a starting estimate. It never posts. You read it and enter your own forecast on your human account.

**Contests**
- **Metaculus Cup** (cash prizes, a new season about every 4 months): the main target. Fall 2026 questions are open now.
- **ACX 2026** ($10k): closed for prizes. The contest snapshot was taken on 17 Jan 2026; later forecasts only count on the site leaderboard. Watch for ACX 2027, which by past pattern should open around December or January (a guess, not confirmed).
- **Hypermind**: I could not confirm a contest that is open right now. Check hypermind.com and read its rules on AI help before entering.

**How to run it (about 5 minutes)**
1. Merge the pull request "Add research briefs for human contests" in your fork.
2. GitHub → Actions → **Research brief for human contests** → **Run workflow**.
   - Leave "links" empty to research all open Metaculus Cup questions, or paste one or more question links (space separated).
   - "Most questions" limits cost: about $0.25–0.50 per question (20 questions ≈ $5–10).
3. When the run finishes (a few minutes per question), open it and download **research-brief** at the bottom. Inside is one Markdown file: per question, the bot's median estimate and each model's reasoning.
4. Read it, check the sources, decide your own number, and enter it on Metaculus with your **human** account. Never use the bot account for human contests.

**Rules to keep**
- Your forecast must be your own judgement. Use the brief as input; don't copy numbers without thinking.
- Keep the old "Forecast on Metaculus Cup" workflow disabled: it posts from the bot account, which can't win prizes and costs money.
- Metaculus pays prizes as cash or a cash equivalent (for example a prepaid card), and winners fill in a tax form (W-8BEN for non-US residents). Prizes can't be sent to US-sanctioned countries; Germany and Indonesia are not on that list.
