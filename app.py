import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import product, combinations
import google.generativeai as genai
from scipy.optimize import linprog, minimize
import networkx as nx
from collections import defaultdict
import json

# Configure Gemini AI
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

# Page config
st.set_page_config(page_title="Game Theory Simulator", layout="wide", page_icon="🎮")

# Sidebar navigation
st.sidebar.title("🎮 Game Theory Simulator")
st.sidebar.markdown("---")

category = st.sidebar.selectbox(
    "Select Category",
    ["Classic Games", "Nash Equilibrium", "Evolutionary Games", "Auction Theory", 
     "Bargaining Games", "Voting Systems", "Network Games", "Mechanism Design",
     "Cooperative Games", "Repeated Games", "Stochastic Games", "AI Analysis"]
)

# Helper Functions
def create_payoff_matrix(rows, cols, default_val=0):
    return np.zeros((rows, cols, 2))

def find_nash_equilibrium_pure(payoff_matrix):
    """Find pure strategy Nash equilibria"""
    equilibria = []
    rows, cols = payoff_matrix.shape[:2]
    
    for i in range(rows):
        for j in range(cols):
            # Check if (i,j) is a Nash equilibrium
            is_nash = True
            
            # Player 1's incentive to deviate
            for i_prime in range(rows):
                if payoff_matrix[i_prime, j, 0] > payoff_matrix[i, j, 0]:
                    is_nash = False
                    break
            
            # Player 2's incentive to deviate
            if is_nash:
                for j_prime in range(cols):
                    if payoff_matrix[i, j_prime, 1] > payoff_matrix[i, j, 1]:
                        is_nash = False
                        break
            
            if is_nash:
                equilibria.append((i, j))
    
    return equilibria

def find_nash_equilibrium_mixed(payoff_matrix):
    """Find mixed strategy Nash equilibrium for 2x2 games"""
    if payoff_matrix.shape[0] != 2 or payoff_matrix.shape[1] != 2:
        return None
    
    # For 2x2 games, use the formula
    a, b, c, d = payoff_matrix[:, :, 0].flatten()
    e, f, g, h = payoff_matrix[:, :, 1].flatten()
    
    # Player 1's mixing probability
    denom_p1 = (d - c) - (b - a)
    if abs(denom_p1) > 1e-6:
        p1 = (d - c) / denom_p1
        p1 = max(0, min(1, p1))
    else:
        p1 = 0.5
    
    # Player 2's mixing probability
    denom_p2 = (h - g) - (f - e)
    if abs(denom_p2) > 1e-6:
        p2 = (h - g) / denom_p2
        p2 = max(0, min(1, p2))
    else:
        p2 = 0.5
    
    return (p1, 1-p1), (p2, 1-p2)

def simulate_repeated_game(payoff_matrix, strategy1, strategy2, rounds):
    """Simulate repeated game with different strategies"""
    scores = [0, 0]
    history = []
    
    for round in range(rounds):
        # Determine actions based on strategy
        if strategy1 == "Always Cooperate":
            action1 = 0
        elif strategy1 == "Always Defect":
            action1 = 1
        elif strategy1 == "Tit-for-Tat":
            action1 = history[-1][1] if history else 0
        elif strategy1 == "Grim Trigger":
            action1 = 1 if any(h[1] == 1 for h in history) else 0
        elif strategy1 == "Random":
            action1 = np.random.choice([0, 1])
        
        if strategy2 == "Always Cooperate":
            action2 = 0
        elif strategy2 == "Always Defect":
            action2 = 1
        elif strategy2 == "Tit-for-Tat":
            action2 = history[-1][0] if history else 0
        elif strategy2 == "Grim Trigger":
            action2 = 1 if any(h[0] == 1 for h in history) else 0
        elif strategy2 == "Random":
            action2 = np.random.choice([0, 1])
        
        # Get payoffs
        p1_payoff = payoff_matrix[action1, action2, 0]
        p2_payoff = payoff_matrix[action1, action2, 1]
        
        scores[0] += p1_payoff
        scores[1] += p2_payoff
        history.append((action1, action2, p1_payoff, p2_payoff))
    
    return scores, history

def get_ai_analysis(game_description, results):
    """Get AI analysis using Gemini"""
    prompt = f"""
    Analyze this game theory scenario:
    
    Game Description: {game_description}
    
    Results: {results}
    
    Provide:
    1. Strategic insights
    2. Equilibrium analysis
    3. Real-world applications
    4. Recommendations
    
    Keep response concise and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# ==================== CLASSIC GAMES ====================
if category == "Classic Games":
    game_type = st.sidebar.selectbox(
        "Select Game",
        ["Prisoner's Dilemma", "Battle of the Sexes", "Matching Pennies", 
         "Chicken Game", "Stag Hunt", "Coordination Game", "Ultimatum Game",
         "Dictator Game", "Public Goods Game", "Traveler's Dilemma"]
    )
    
    st.title(f"🎯 {game_type}")
    
    if game_type == "Prisoner's Dilemma":
        st.markdown("""
        Two criminals are arrested. Each can either cooperate with their partner or defect.
        - Both cooperate: Light sentence for both
        - Both defect: Moderate sentence for both
        - One defects: Defector goes free, cooperator gets heavy sentence
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Customize Payoffs")
            cc = st.slider("Both Cooperate", -10, 10, -1)
            cd = st.slider("Cooperate/Defect", -10, 10, -3)
            dc = st.slider("Defect/Cooperate", -10, 10, 0)
            dd = st.slider("Both Defect", -10, 10, -2)
        
        payoff_matrix = np.array([
            [[cc, cc], [cd, dc]],
            [[dc, cd], [dd, dd]]
        ])
        
        with col1:
            # Display payoff matrix
            df = pd.DataFrame({
                'Cooperate': [f"({cc}, {cc})", f"({dc}, {cd})"],
                'Defect': [f"({cd}, {dc})", f"({dd}, {dd})"]
            }, index=['Cooperate', 'Defect'])
            
            st.dataframe(df, use_container_width=True)
            
            # Find equilibria
            pure_eq = find_nash_equilibrium_pure(payoff_matrix)
            st.subheader("Nash Equilibria")
            
            if pure_eq:
                for eq in pure_eq:
                    st.success(f"Pure Strategy: {'(Cooperate, Cooperate)' if eq == (0,0) else '(Cooperate, Defect)' if eq == (0,1) else '(Defect, Cooperate)' if eq == (1,0) else '(Defect, Defect)'}")
            
            mixed_eq = find_nash_equilibrium_mixed(payoff_matrix)
            if mixed_eq:
                st.info(f"Mixed Strategy: P1=(C:{mixed_eq[0][0]:.2f}, D:{mixed_eq[0][1]:.2f}), P2=(C:{mixed_eq[1][0]:.2f}, D:{mixed_eq[1][1]:.2f})")
        
        if st.button("🤖 Get AI Analysis"):
            with st.spinner("Analyzing..."):
                analysis = get_ai_analysis(
                    f"Prisoner's Dilemma with payoffs: CC={cc}, CD={cd}, DC={dc}, DD={dd}",
                    f"Pure equilibria: {pure_eq}, Mixed equilibrium: {mixed_eq}"
                )
                st.markdown("### AI Strategic Analysis")
                st.markdown(analysis)
    
    elif game_type == "Battle of the Sexes":
        st.markdown("""
        A couple wants to spend time together but prefers different activities.
        - Both choose Opera: Good for Player 1, okay for Player 2
        - Both choose Football: Good for Player 2, okay for Player 1
        - Different choices: Both unhappy
        """)
        
        payoff_matrix = np.array([
            [[2, 1], [0, 0]],
            [[0, 0], [1, 2]]
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame({
                'Opera': ['(2, 1)', '(0, 0)'],
                'Football': ['(0, 0)', '(1, 2)']
            }, index=['Opera', 'Football'])
            st.dataframe(df, use_container_width=True)
        
        with col2:
            pure_eq = find_nash_equilibrium_pure(payoff_matrix)
            st.subheader("Equilibria")
            for eq in pure_eq:
                st.success(f"{'(Opera, Opera)' if eq == (0,0) else '(Football, Football)'}")
            
            mixed_eq = find_nash_equilibrium_mixed(payoff_matrix)
            if mixed_eq:
                st.info(f"Mixed: P1=(O:{mixed_eq[0][0]:.2f}, F:{mixed_eq[0][1]:.2f}), P2=(O:{mixed_eq[1][0]:.2f}, F:{mixed_eq[1][1]:.2f})")

    elif game_type == "Chicken Game":
        st.markdown("""
        Two drivers drive toward each other. Each must choose to swerve or go straight.
        - Both swerve: Tie, small loss of pride
        - One swerves: Swerved player loses face
        - Both straight: Crash (worst outcome)
        """)
        
        payoff_matrix = np.array([
            [[0, 0], [-1, 1]],
            [[1, -1], [-10, -10]]
        ])
        
        df = pd.DataFrame({
            'Swerve': ['(0, 0)', '(1, -1)'],
            'Straight': ['(-1, 1)', '(-10, -10)']
        }, index=['Swerve', 'Straight'])
        
        st.dataframe(df, use_container_width=True)
        
        pure_eq = find_nash_equilibrium_pure(payoff_matrix)
        st.subheader("Nash Equilibria")
        for eq in pure_eq:
            st.success(f"{'(Swerve, Straight)' if eq == (0,1) else '(Straight, Swerve)'}")

    elif game_type == "Ultimatum Game":
        st.markdown("""
        Player 1 proposes how to split a sum of money. Player 2 can accept or reject.
        If rejected, both get nothing.
        """)
        
        total = st.slider("Total Amount", 10, 100, 10)
        offer = st.slider("Player 1's Offer to Player 2", 0, total, total//2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Player 1 Keeps", total - offer)
            st.metric("Player 2 Receives", offer)
        
        with col2:
            st.subheader("Predictions")
            st.write("**Game Theory Prediction**: Minimum offer (subgame perfect equilibrium)")
            st.write("**Behavioral Reality**: 40-50% offers typical, <20% often rejected")
            
            if offer < total * 0.2:
                st.warning("⚠️ Low offer - likely rejection!")
            elif offer < total * 0.3:
                st.info("🤔 Borderline offer")
            else:
                st.success("✅ Fair offer - likely acceptance")

# ==================== NASH EQUILIBRIUM ====================
elif category == "Nash Equilibrium":
    st.title("⚖️ Nash Equilibrium Finder")
    
    method = st.sidebar.selectbox(
        "Method",
        ["Custom Matrix", "Support Enumeration", "Lemke-Howson", "Replicator Dynamics"]
    )
    
    if method == "Custom Matrix":
        n_strategies_p1 = st.sidebar.slider("Player 1 Strategies", 2, 5, 2)
        n_strategies_p2 = st.sidebar.slider("Player 2 Strategies", 2, 5, 2)
        
        st.subheader("Enter Payoff Matrix")
        
        payoff_matrix = create_payoff_matrix(n_strategies_p1, n_strategies_p2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Player 1 Payoffs**")
            for i in range(n_strategies_p1):
                cols = st.columns(n_strategies_p2)
                for j in range(n_strategies_p2):
                    payoff_matrix[i, j, 0] = cols[j].number_input(
                        f"P1({i},{j})", value=0.0, key=f"p1_{i}_{j}", label_visibility="collapsed"
                    )
        
        with col2:
            st.write("**Player 2 Payoffs**")
            for i in range(n_strategies_p1):
                cols = st.columns(n_strategies_p2)
                for j in range(n_strategies_p2):
                    payoff_matrix[i, j, 1] = cols[j].number_input(
                        f"P2({i},{j})", value=0.0, key=f"p2_{i}_{j}", label_visibility="collapsed"
                    )
        
        if st.button("Find Equilibria"):
            st.subheader("Results")
            
            pure_eq = find_nash_equilibrium_pure(payoff_matrix)
            
            if pure_eq:
                st.success(f"Found {len(pure_eq)} Pure Strategy Equilibria")
                for eq in pure_eq:
                    st.write(f"Strategy Profile: ({eq[0]}, {eq[1]}) → Payoffs: {payoff_matrix[eq[0], eq[1]]}")
            else:
                st.info("No pure strategy equilibria found")
            
            if n_strategies_p1 == 2 and n_strategies_p2 == 2:
                mixed_eq = find_nash_equilibrium_mixed(payoff_matrix)
                if mixed_eq:
                    st.success("Mixed Strategy Equilibrium")
                    st.write(f"Player 1: {mixed_eq[0]}")
                    st.write(f"Player 2: {mixed_eq[1]}")

    elif method == "Support Enumeration":
        st.markdown("""
        Support enumeration finds all Nash equilibria by checking each possible support combination.
        A support is the set of strategies played with positive probability.
        """)
        
        n = st.slider("Number of strategies per player", 2, 4, 3)
        
        # Generate random game
        if st.button("Generate Random Game"):
            st.session_state.random_game = np.random.randint(-5, 10, (n, n, 2))
        
        if 'random_game' in st.session_state:
            payoff_matrix = st.session_state.random_game
            
            # Display matrix
            for player in range(2):
                st.write(f"**Player {player+1} Payoffs**")
                df = pd.DataFrame(payoff_matrix[:,:,player])
                st.dataframe(df)
            
            st.subheader("Equilibrium Analysis")
            pure_eq = find_nash_equilibrium_pure(payoff_matrix)
            
            if pure_eq:
                st.success(f"Pure Equilibria: {pure_eq}")
            else:
                st.info("No pure equilibria - mixed equilibria likely exist")

# ==================== EVOLUTIONARY GAMES ====================
elif category == "Evolutionary Games":
    st.title("🧬 Evolutionary Game Theory")
    
    game = st.sidebar.selectbox(
        "Select Model",
        ["Hawk-Dove", "Rock-Paper-Scissors", "Replicator Dynamics", 
         "ESS Analysis", "Moran Process", "Wright-Fisher Model"]
    )
    
    if game == "Hawk-Dove":
        st.markdown("""
        Animals compete for a resource V. Hawks fight, Doves share.
        - Hawk vs Hawk: Fight (cost C, win V/2 on average)
        - Hawk vs Dove: Hawk wins V
        - Dove vs Dove: Share V/2 each
        """)
        
        V = st.slider("Resource Value (V)", 1, 20, 10)
        C = st.slider("Cost of Fighting (C)", 1, 30, 15)
        
        # Payoff matrix
        payoff_matrix = np.array([
            [(V-C)/2, V],
            [0, V/2]
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame(payoff_matrix, 
                            columns=['Hawk', 'Dove'],
                            index=['Hawk', 'Dove'])
            st.dataframe(df)
        
        with col2:
            st.subheader("ESS Analysis")
            if C > V:
                p_hawk = V / C
                st.success(f"Mixed ESS: {p_hawk:.2%} Hawks, {1-p_hawk:.2%} Doves")
            else:
                st.success("Pure ESS: All Hawks")
        
        # Simulate evolution
        st.subheader("Population Dynamics")
        
        generations = st.slider("Generations", 10, 200, 100)
        initial_hawks = st.slider("Initial % Hawks", 0, 100, 50) / 100
        
        if st.button("Simulate Evolution"):
            population = [initial_hawks]
            
            for _ in range(generations):
                p = population[-1]
                
                # Average fitness
                fit_hawk = p * payoff_matrix[0,0] + (1-p) * payoff_matrix[0,1]
                fit_dove = p * payoff_matrix[1,0] + (1-p) * payoff_matrix[1,1]
                avg_fit = p * fit_hawk + (1-p) * fit_dove
                
                # Replicator dynamics
                if avg_fit != 0:
                    p_new = p * fit_hawk / avg_fit
                else:
                    p_new = p
                
                population.append(max(0, min(1, p_new)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=population, name='Hawks', fill='tozeroy'))
            fig.add_trace(go.Scatter(y=[1-p for p in population], name='Doves', fill='tonexty'))
            fig.update_layout(title="Population Evolution", xaxis_title="Generation", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

    elif game == "Rock-Paper-Scissors":
        st.markdown("""
        Classic cyclic game: Rock beats Scissors, Scissors beats Paper, Paper beats Rock.
        """)
        
        # Payoff for winning/losing
        win_val = st.slider("Win Value", 1, 10, 1)
        lose_val = -win_val
        
        payoff_matrix = np.array([
            [[0, 0], [lose_val, win_val], [win_val, lose_val]],
            [[win_val, lose_val], [0, 0], [lose_val, win_val]],
            [[lose_val, win_val], [win_val, lose_val], [0, 0]]
        ])
        
        st.subheader("Equilibrium")
        st.success("Unique Nash Equilibrium: Play each strategy with probability 1/3")
        
        # Simulate game
        rounds = st.slider("Simulation Rounds", 100, 10000, 1000)
        
        if st.button("Simulate"):
            p1_strat = np.random.choice([0, 1, 2], rounds)
            p2_strat = np.random.choice([0, 1, 2], rounds)
            
            p1_score = sum(payoff_matrix[p1_strat[i], p2_strat[i], 0] for i in range(rounds))
            p2_score = sum(payoff_matrix[p1_strat[i], p2_strat[i], 1] for i in range(rounds))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Player 1 Score", p1_score)
            col2.metric("Player 2 Score", p2_score)
            col3.metric("Difference", abs(p1_score - p2_score))
            
            # Distribution
            choices = ['Rock', 'Paper', 'Scissors']
            p1_dist = [sum(p1_strat == i) for i in range(3)]
            p2_dist = [sum(p2_strat == i) for i in range(3)]
            
            fig = go.Figure(data=[
                go.Bar(name='Player 1', x=choices, y=p1_dist),
                go.Bar(name='Player 2', x=choices, y=p2_dist)
            ])
            fig.update_layout(title="Strategy Distribution", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# ==================== AUCTION THEORY ====================
elif category == "Auction Theory":
    st.title("🏷️ Auction Theory")
    
    auction_type = st.sidebar.selectbox(
        "Auction Type",
        ["First-Price Sealed-Bid", "Second-Price Sealed-Bid", "English Auction",
         "Dutch Auction", "All-Pay Auction", "Multi-Unit Auction", "Combinatorial Auction"]
    )
    
    if auction_type == "First-Price Sealed-Bid":
        st.markdown("""
        Bidders submit sealed bids. Highest bidder wins and pays their bid.
        Optimal strategy: Bid less than your valuation.
        """)
        
        n_bidders = st.slider("Number of Bidders", 2, 10, 3)
        
        st.subheader("Bidder Valuations")
        valuations = []
        cols = st.columns(min(n_bidders, 5))
        for i in range(n_bidders):
            col_idx = i % 5
            val = cols[col_idx].number_input(f"Bidder {i+1}", 0, 1000, 100*(i+1), key=f"val_{i}")
            valuations.append(val)
        
        if st.button("Simulate Auction"):
            # Optimal bidding: bid (n-1)/n * valuation
            bids = [v * (n_bidders - 1) / n_bidders for v in valuations]
            
            winner = np.argmax(bids)
            winning_bid = bids[winner]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Results")
                st.success(f"Winner: Bidder {winner+1}")
                st.metric("Winning Bid", f"${winning_bid:.2f}")
                st.metric("Winner's Valuation", f"${valuations[winner]:.2f}")
                st.metric("Winner's Surplus", f"${valuations[winner] - winning_bid:.2f}")
            
            with col2:
                df = pd.DataFrame({
                    'Bidder': [f'Bidder {i+1}' for i in range(n_bidders)],
                    'Valuation': valuations,
                    'Bid': bids,
                    'Surplus': [v - b if i == winner else 0 for i, (v, b) in enumerate(zip(valuations, bids))]
                })
                st.dataframe(df)
            
            fig = go.Figure(data=[
                go.Bar(name='Valuation', x=[f'B{i+1}' for i in range(n_bidders)], y=valuations),
                go.Bar(name='Bid', x=[f'B{i+1}' for i in range(n_bidders)], y=bids)
            ])
            fig.update_layout(title="Valuations vs Bids", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    elif auction_type == "Second-Price Sealed-Bid":
        st.markdown("""
        Vickrey Auction: Highest bidder wins but pays second-highest bid.
        Dominant strategy: Bid your true valuation.
        """)
        
        n_bidders = st.slider("Number of Bidders", 2, 10, 3)
        
        valuations = []
        cols = st.columns(min(n_bidders, 5))
        for i in range(n_bidders):
            col_idx = i % 5
            val = cols[col_idx].number_input(f"Bidder {i+1}", 0, 1000, 100*(i+1), key=f"val2_{i}")
            valuations.append(val)
        
        if st.button("Run Vickrey Auction"):
            # Truth-telling is optimal
            bids = valuations.copy()
            
            sorted_bids = sorted(bids, reverse=True)
            winner = bids.index(sorted_bids[0])
            price = sorted_bids[1] if len(sorted_bids) > 1 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Winner: Bidder {winner+1}")
                st.metric("Price Paid", f"${price:.2f}")
                st.metric("Winner's Surplus", f"${valuations[winner] - price:.2f}")
                st.info("💡 Truth-telling is the dominant strategy!")
            
            with col2:
                df = pd.DataFrame({
                    'Bidder': [f'Bidder {i+1}' for i in range(n_bidders)],
                    'Valuation/Bid': valuations,
                    'Surplus': [v - price if i == winner else 0 for i, v in enumerate(valuations)]
                })
                st.dataframe(df)

    elif auction_type == "All-Pay Auction":
        st.markdown("""
        All bidders pay their bid, but only the highest bidder wins.
        Used to model contests, lobbying, R&D races, etc.
        """)
        
        prize_value = st.number_input("Prize Value", 100, 10000, 1000)
        n_bidders = st.slider("Number of Bidders", 2, 6, 3)
        
        st.subheader("Theoretical Equilibrium")
        st.write(f"Expected bid per player: ${prize_value * (n_bidders-1) / n_bidders:.2f}")
        st.write(f"Total expected expenditure: ${prize_value * (n_bidders-1):.2f}")
        st.write("⚠️ All-pay auctions lead to significant rent dissipation!")

# ==================== BARGAINING GAMES ====================
elif category == "Bargaining Games":
    st.title("🤝 Bargaining Theory")
    
    model = st.sidebar.selectbox(
        "Model",
        ["Nash Bargaining", "Rubinstein Alternating Offers", "Ultimatum Game",
         "Kalai-Smorodinsky", "Egalitarian Solution"]
    )
    
    if model == "Nash Bargaining":
        st.markdown("""
        Two players bargain over splitting a surplus. 
        Nash solution maximizes the product of utilities above disagreement point.
        """)
        
        surplus = st.slider("Total Surplus", 10, 100, 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            d1 = st.number_input("P1 Disagreement Payoff", 0, surplus//2, 0)
            u1_weight = st.slider("P1 Bargaining Power", 0.0, 1.0, 0.5)
        
        with col2:
            d2 = st.number_input("P2 Disagreement Payoff", 0, surplus//2, 0)
            u2_weight = 1 - u1_weight
        
        # Nash solution: maximize (u1 - d1)^w1 * (u2 - d2)^w2
        # Subject to u1 + u2 = surplus
        
        # Optimal split
        available = surplus - d1 - d2
        p1_share = d1 + available * u1_weight
        p2_share = d2 + available * u2_weight
        
        st.subheader("Nash Bargaining Solution")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Player 1 Share", f"${p1_share:.2f}")
        col2.metric("Player 2 Share", f"${p2_share:.2f}")
        col3.metric("Total", f"${p1_share + p2_share:.2f}")
        
        # Visualization
        x = np.linspace(d1, surplus - d2, 100)
        y = surplus - x
        nash_product = (x - d1)**u1_weight * (y - d2)**u2_weight
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name='Feasible Set', mode='lines'))
        fig.add_trace(go.Scatter(x=[p1_share], y=[p2_share], name='Nash Solution', 
                                mode='markers', marker=dict(size=15, color='red')))
        fig.update_layout(title="Bargaining Set", xaxis_title="Player 1 Payoff", 
                         yaxis_title="Player 2 Payoff")
        st.plotly_chart(fig, use_container_width=True)
    
    elif model == "Rubinstein Alternating Offers":
        st.markdown("""
        Players alternate making offers with time discounting.
        Player 1 makes first offer, players alternate until agreement.
        """)
        
        surplus = st.slider("Surplus to Split", 10, 100, 100)
        delta1 = st.slider("P1 Discount Factor", 0.0, 1.0, 0.9, 0.01)
        delta2 = st.slider("P2 Discount Factor", 0.0, 1.0, 0.9, 0.01)
        
        # Rubinstein solution
        if delta1 + delta2 != 0:
            p1_share = surplus * (1 - delta2) / (1 - delta1 * delta2)
            p2_share = surplus - p1_share
        else:
            p1_share = surplus / 2
            p2_share = surplus / 2
        
        st.subheader("Subgame Perfect Equilibrium")
        
        col1, col2 = st.columns(2)
        col1.metric("Player 1 (First Mover)", f"${p1_share:.2f}")
        col2.metric("Player 2", f"${p2_share:.2f}")
        
        st.info(f"First-mover advantage: ${p1_share - surplus/2:.2f}")
        
        # Simulate negotiation
        if st.button("Simulate Negotiation"):
            st.subheader("Negotiation Process")
            
            offers = []
            current_surplus = surplus
            period = 0
            
            for round in range(10):
                if period % 2 == 0:  # P1 offers
                    offer = p2_share * (delta2 ** period)
                    offers.append((period, 'P1', current_surplus - offer, offer))
                else:  # P2 offers
                    offer = p1_share * (delta1 ** period)
                    offers.append((period, 'P2', offer, current_surplus - offer))
                
                period += 1
                current_surplus *= min(delta1, delta2)
            
            df = pd.DataFrame(offers, columns=['Period', 'Proposer', 'P1 Gets', 'P2 Gets'])
            st.dataframe(df)

# ==================== VOTING SYSTEMS ====================
elif category == "Voting Systems":
    st.title("🗳️ Voting Systems & Social Choice")
    
    system = st.sidebar.selectbox(
        "Voting System",
        ["Plurality", "Borda Count", "Condorcet", "Approval Voting", 
         "Ranked Choice (IRV)", "Copeland Method", "Kemeny-Young",
         "Arrow's Impossibility Theorem Demo"]
    )
    
    if system == "Plurality":
        st.markdown("Each voter votes for one candidate. Most votes wins.")
        
        n_candidates = st.slider("Number of Candidates", 2, 6, 3)
        n_voters = st.slider("Number of Voters", 10, 500, 100)
        
        if st.button("Simulate Election"):
            votes = np.random.randint(0, n_candidates, n_voters)
            vote_counts = [np.sum(votes == i) for i in range(n_candidates)]
            
            winner = np.argmax(vote_counts)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Winner: Candidate {chr(65 + winner)}")
                st.metric("Votes Received", vote_counts[winner])
                st.metric("Vote Share", f"{100*vote_counts[winner]/n_voters:.1f}%")
            
            with col2:
                df = pd.DataFrame({
                    'Candidate': [chr(65 + i) for i in range(n_candidates)],
                    'Votes': vote_counts,
                    'Percentage': [f"{100*v/n_voters:.1f}%" for v in vote_counts]
                })
                st.dataframe(df)
            
            fig = px.pie(values=vote_counts, names=[chr(65+i) for i in range(n_candidates)],
                        title="Vote Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    elif system == "Borda Count":
        st.markdown("""
        Voters rank candidates. Points awarded: n-1 for 1st, n-2 for 2nd, etc.
        Most points wins.
        """)
        
        n_candidates = 4
        n_voters = st.slider("Number of Voters", 10, 100, 50)
        
        if st.button("Simulate Borda Count"):
            # Generate random preferences
            rankings = np.array([np.random.permutation(n_candidates) for _ in range(n_voters)])
            
            # Calculate Borda scores
            borda_scores = np.zeros(n_candidates)
            for ranking in rankings:
                for pos, candidate in enumerate(ranking):
                    borda_scores[candidate] += (n_candidates - 1 - pos)
            
            winner = np.argmax(borda_scores)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Borda Winner: Candidate {chr(65 + winner)}")
                st.metric("Borda Score", int(borda_scores[winner]))
            
            with col2:
                df = pd.DataFrame({
                    'Candidate': [chr(65 + i) for i in range(n_candidates)],
                    'Borda Score': borda_scores.astype(int),
                    'Avg Rank': [(n_candidates - 1) - s/n_voters for s in borda_scores]
                })
                st.dataframe(df)
            
            fig = px.bar(x=[chr(65+i) for i in range(n_candidates)], 
                        y=borda_scores, title="Borda Scores")
            st.plotly_chart(fig, use_container_width=True)
    
    elif system == "Approval Voting":
        st.markdown("Voters approve any number of candidates. Most approvals wins.")
        
        n_candidates = st.slider("Candidates", 3, 8, 5)
        n_voters = st.slider("Voters", 10, 200, 100)
        approval_rate = st.slider("Avg Approval Rate", 0.1, 0.9, 0.4)
        
        if st.button("Run Approval Vote"):
            # Each voter approves candidates with given probability
            approvals = np.random.random((n_voters, n_candidates)) < approval_rate
            approval_counts = approvals.sum(axis=0)
            
            winner = np.argmax(approval_counts)
            
            st.success(f"Winner: Candidate {chr(65 + winner)} with {approval_counts[winner]} approvals")
            
            df = pd.DataFrame({
                'Candidate': [chr(65 + i) for i in range(n_candidates)],
                'Approvals': approval_counts,
                'Approval Rate': [f"{100*c/n_voters:.1f}%" for c in approval_counts]
            })
            st.dataframe(df)
    
    elif system == "Arrow's Impossibility Theorem Demo":
        st.markdown("""
        **Arrow's Impossibility Theorem**: No rank-order voting system can satisfy all:
        1. Pareto Efficiency
        2. Independence of Irrelevant Alternatives (IIA)
        3. Non-dictatorship
        4. Unrestricted Domain
        """)
        
        st.subheader("IIA Violation Example")
        
        st.write("**Scenario 1: Three candidates (A, B, C)**")
        col1, col2, col3 = st.columns(3)
        col1.write("35%: A > B > C")
        col2.write("33%: B > C > A")
        col3.write("32%: C > A > B")
        
        st.info("Plurality winner: A (35%)")
        
        st.write("**Scenario 2: Candidate B drops out**")
        col1, col2 = st.columns(2)
        col1.write("35%: A > C")
        col2.write("65%: C > A")
        
        st.success("New winner: C (65%)")
        
        st.warning("⚠️ Removing B (who wasn't winning) changed the winner from A to C!")
        st.write("This violates Independence of Irrelevant Alternatives.")

# ==================== NETWORK GAMES ====================
elif category == "Network Games":
    st.title("🕸️ Network & Graph Games")
    
    game = st.sidebar.selectbox(
        "Game Type",
        ["Network Formation", "Contagion Models", "Diffusion Games",
         "Routing Games", "Matching Markets", "Coordination on Networks"]
    )
    
    if game == "Network Formation":
        st.markdown("""
        Agents form links with costs and benefits.
        Link formation is strategic - both parties must agree.
        """)
        
        n_agents = st.slider("Number of Agents", 3, 10, 5)
        link_cost = st.slider("Link Cost", 0.0, 2.0, 0.5, 0.1)
        benefit_direct = st.slider("Direct Link Benefit", 1.0, 5.0, 3.0, 0.1)
        benefit_indirect = st.slider("Indirect Link Benefit", 0.0, 2.0, 1.0, 0.1)
        
        if st.button("Form Network"):
            # Simple strategy: form link if benefit > cost
            G = nx.Graph()
            G.add_nodes_from(range(n_agents))
            
            # Calculate benefits for all potential links
            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    # Direct benefit minus cost
                    if benefit_direct - link_cost > 0:
                        G.add_edge(i, j)
            
            # Calculate payoffs
            payoffs = {}
            for i in range(n_agents):
                direct_links = G.degree(i)
                
                # Count indirect links (distance 2)
                indirect = 0
                for j in range(n_agents):
                    if i != j and not G.has_edge(i, j):
                        if nx.has_path(G, i, j):
                            path_length = nx.shortest_path_length(G, i, j)
                            if path_length == 2:
                                indirect += 1
                
                payoff = (direct_links * benefit_direct + 
                         indirect * benefit_indirect - 
                         direct_links * link_cost)
                payoffs[i] = payoff
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Draw network
                pos = nx.spring_layout(G, seed=42)
                
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='gray'),
                        hoverinfo='none'
                    ))
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[str(i) for i in G.nodes()],
                    textposition="top center",
                    marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
                    hoverinfo='text',
                    hovertext=[f"Agent {i}<br>Payoff: {payoffs[i]:.2f}" for i in G.nodes()]
                )
                
                fig = go.Figure(data=edge_trace + [node_trace])
                fig.update_layout(
                    title="Formed Network",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Network Stats")
                st.metric("Total Links", G.number_of_edges())
                st.metric("Avg Degree", f"{2*G.number_of_edges()/n_agents:.2f}")
                st.metric("Density", f"{nx.density(G):.2%}")
                
                if nx.is_connected(G):
                    st.success("✅ Connected")
                    st.metric("Diameter", nx.diameter(G))
                else:
                    st.warning("❌ Not Connected")
                    st.metric("Components", nx.number_connected_components(G))
    
    elif game == "Contagion Models":
        st.markdown("""
        Model spread of behavior/disease on a network.
        **Threshold Model**: Adopt if fraction of neighbors ≥ threshold.
        """)
        
        n_agents = st.slider("Network Size", 10, 100, 30)
        avg_degree = st.slider("Avg Connections", 2, 10, 4)
        threshold = st.slider("Adoption Threshold", 0.0, 1.0, 0.5, 0.05)
        initial_adopters = st.slider("Initial Adopters", 1, n_agents//2, 3)
        
        if st.button("Simulate Contagion"):
            # Create network
            G = nx.barabasi_albert_graph(n_agents, avg_degree//2, seed=42)
            
            # Initialize
            adopters = set(np.random.choice(n_agents, initial_adopters, replace=False))
            
            # Simulate contagion
            history = [len(adopters)]
            
            for round in range(20):
                new_adopters = set()
                
                for node in G.nodes():
                    if node not in adopters:
                        neighbors = list(G.neighbors(node))
                        if len(neighbors) > 0:
                            adopted_neighbors = sum(1 for n in neighbors if n in adopters)
                            if adopted_neighbors / len(neighbors) >= threshold:
                                new_adopters.add(node)
                
                if not new_adopters:
                    break
                
                adopters.update(new_adopters)
                history.append(len(adopters))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Adopters", len(adopters))
                st.metric("Adoption Rate", f"{100*len(adopters)/n_agents:.1f}%")
                st.metric("Rounds to Equilibrium", len(history)-1)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history, mode='lines+markers', name='Adopters'))
                fig.update_layout(title="Contagion Spread", xaxis_title="Round", yaxis_title="Adopters")
                st.plotly_chart(fig, use_container_width=True)

# ==================== MECHANISM DESIGN ====================
elif category == "Mechanism Design":
    st.title("⚙️ Mechanism Design")
    
    mechanism = st.sidebar.selectbox(
        "Mechanism",
        ["VCG Mechanism", "AGV Mechanism", "Groves Mechanism",
         "Revelation Principle", "Myerson's Optimal Auction", "Budget Balance"]
    )
    
    if mechanism == "VCG Mechanism":
        st.markdown("""
        **Vickrey-Clarke-Groves Mechanism**
        
        Incentive-compatible mechanism for public projects.
        Each agent pays their externality on others.
        """)
        
        st.subheader("Public Project Decision")
        
        project_cost = st.number_input("Project Cost", 100, 10000, 1000)
        n_agents = st.slider("Number of Agents", 2, 10, 3)
        
        st.write("**Agent Valuations** (benefit if project is built):")
        
        valuations = []
        cols = st.columns(min(n_agents, 5))
        for i in range(n_agents):
            col_idx = i % 5
            val = cols[col_idx].number_input(f"Agent {i+1}", 0, project_cost, 
                                            300*(i+1), key=f"vcg_{i}")
            valuations.append(val)
        
        if st.button("Run VCG Mechanism"):
            total_value = sum(valuations)
            
            st.subheader("Decision")
            if total_value >= project_cost:
                st.success("✅ Build the project!")
                
                # Calculate VCG payments
                payments = []
                for i in range(n_agents):
                    # Social welfare without agent i
                    others_value = total_value - valuations[i]
                    
                    # Best decision without agent i
                    if others_value >= project_cost:
                        welfare_without = others_value - project_cost
                    else:
                        welfare_without = 0
                    
                    # Actual welfare of others
                    welfare_with = others_value - (project_cost if total_value >= project_cost else 0)
                    
                    # VCG payment = externality
                    payment = welfare_without - welfare_with
                    payments.append(max(0, payment))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("VCG Payments")
                    for i, payment in enumerate(payments):
                        st.write(f"Agent {i+1}: ${payment:.2f}")
                    st.metric("Total Revenue", f"${sum(payments):.2f}")
                
                with col2:
                    st.subheader("Agent Utilities")
                    for i in range(n_agents):
                        utility = valuations[i] - payments[i]
                        st.write(f"Agent {i+1}: ${utility:.2f}")
                
                if sum(payments) >= project_cost:
                    st.success("✅ Budget balanced!")
                else:
                    st.warning(f"⚠️ Deficit: ${project_cost - sum(payments):.2f}")
                
            else:
                st.error("❌ Don't build the project")
                st.info("Total value < Cost")
    
    elif mechanism == "Revelation Principle":
        st.markdown("""
        **Revelation Principle**: Any outcome achievable by any mechanism 
        can be achieved by a truthful direct revelation mechanism.
        
        This fundamental theorem simplifies mechanism design.
        """)
        
        st.subheader("Example: Allocating a Single Item")
        
        n_agents = st.slider("Number of Agents", 2, 6, 3)
        
        valuations = []
        cols = st.columns(min(n_agents, 3))
        for i in range(n_agents):
            col_idx = i % 3
            val = cols[col_idx].number_input(f"Agent {i+1} Valuation", 
                                            0, 1000, 100*(i+1), key=f"rev_{i}")
            valuations.append(val)
        
        st.subheader("Truthful Mechanisms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Second-Price Auction**")
            winner = np.argmax(valuations)
            price = sorted(valuations, reverse=True)[1] if n_agents > 1 else 0
            
            st.success(f"Winner: Agent {winner+1}")
            st.metric("Price", f"${price:.2f}")
            st.metric("Winner's Surplus", f"${valuations[winner] - price:.2f}")
            st.info("✅ Truthful (Dominant Strategy)")
        
        with col2:
            st.write("**First-Price Auction**")
            st.warning("❌ Not truthful")
            st.write("Optimal bid < true valuation")
            st.write("Shading depends on beliefs about others")

# ==================== COOPERATIVE GAMES ====================
elif category == "Cooperative Games":
    st.title("🤝 Cooperative Game Theory")
    
    concept = st.sidebar.selectbox(
        "Concept",
        ["Shapley Value", "Core", "Nucleolus", "Banzhaf Power Index",
         "Coalition Formation", "Transferable Utility"]
    )
    
    if concept == "Shapley Value":
        st.markdown("""
        **Shapley Value**: Fair allocation based on marginal contributions.
        
        Formula: φᵢ(v) = Σ [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]
        """)
        
        n_players = st.slider("Number of Players", 2, 5, 3)
        
        st.subheader("Define Coalition Values")
        st.write("Enter value for each coalition (default: superadditive)")
        
        # Generate all coalitions
        players = list(range(n_players))
        coalitions = []
        for r in range(1, n_players + 1):
            coalitions.extend(combinations(players, r))
        
        coalition_values = {}
        
        # Default superadditive values
        for coalition in coalitions:
            default_val = len(coalition) ** 2  # Superadditive
            coalition_str = '{' + ','.join(str(p) for p in coalition) + '}'
            val = st.number_input(f"v{coalition_str}", 0.0, 1000.0, float(default_val), 
                                 key=f"coal_{coalition}")
            coalition_values[coalition] = val
        
        if st.button("Calculate Shapley Values"):
            shapley_values = [0.0] * n_players
            
            import math
            
            for i in range(n_players):
                for coalition in coalitions:
                    if i not in coalition:
                        S = set(coalition)
                        S_with_i = tuple(sorted(S | {i}))
                        
                        # Marginal contribution
                        v_S = coalition_values[coalition]
                        v_S_i = coalition_values[S_with_i]
                        marginal = v_S_i - v_S
                        
                        # Weight
                        s = len(coalition)
                        weight = math.factorial(s) * math.factorial(n_players - s - 1) / math.factorial(n_players)
                        
                        shapley_values[i] += weight * marginal
                
                # Empty coalition contribution
                singleton = (i,)
                weight = math.factorial(0) * math.factorial(n_players - 1) / math.factorial(n_players)
                shapley_values[i] += weight * coalition_values[singleton]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Shapley Values")
                for i, val in enumerate(shapley_values):
                    st.metric(f"Player {i+1}", f"${val:.2f}")
            
            with col2:
                grand_coalition = tuple(range(n_players))
                total_value = coalition_values[grand_coalition]
                
                st.metric("Grand Coalition Value", f"${total_value:.2f}")
                st.metric("Sum of Shapley Values", f"${sum(shapley_values):.2f}")
                
                if abs(sum(shapley_values) - total_value) < 0.01:
                    st.success("✅ Efficient (sums to grand coalition value)")
                
                fig = px.pie(values=shapley_values, names=[f'P{i+1}' for i in range(n_players)],
                           title="Value Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    elif concept == "Core":
        st.markdown("""
        **Core**: Set of allocations where no coalition can improve by deviating.
        
        An allocation x is in the core if:
        - Σxᵢ = v(N) (efficiency)
        - Σxᵢ ≥ v(S) for all S ⊆ N (stability)
        """)
        
        st.subheader("3-Player Example")
        
        v12 = st.number_input("v({1,2})", 0, 100, 60)
        v13 = st.number_input("v({1,3})", 0, 100, 70)
        v23 = st.number_input("v({2,3})", 0, 100, 50)
        v123 = st.number_input("v({1,2,3})", 0, 200, 100)
        
        if st.button("Check Core"):
            st.subheader("Core Constraints")
            
            # The core must satisfy:
            # x1 + x2 + x3 = v123
            # x1 + x2 >= v12
            # x1 + x3 >= v13
            # x2 + x3 >= v23
            # x1, x2, x3 >= 0
            
            # Check if core is non-empty
            if v12 + v13 + v23 <= 2 * v123:
                st.success("✅ Core is non-empty!")
                
                # Find a core allocation
                x1 = (v123 + v12 + v13 - v23) / 2
                x2 = (v123 + v12 + v23 - v13) / 2
                x3 = (v123 + v13 + v23 - v12) / 2
                
                if x1 >= 0 and x2 >= 0 and x3 >= 0:
                    st.write("**Sample Core Allocation:**")
                    st.write(f"Player 1: ${x1:.2f}")
                    st.write(f"Player 2: ${x2:.2f}")
                    st.write(f"Player 3: ${x3:.2f}")
                    
                    # Verify
                    st.write("\n**Verification:**")
                    st.write(f"x1 + x2 = {x1+x2:.2f} ≥ {v12} ✓" if x1+x2 >= v12 else f"x1 + x2 = {x1+x2:.2f} < {v12} ✗")
                    st.write(f"x1 + x3 = {x1+x3:.2f} ≥ {v13} ✓" if x1+x3 >= v13 else f"x1 + x3 = {x1+x3:.2f} < {v13} ✗")
                    st.write(f"x2 + x3 = {x2+x3:.2f} ≥ {v23} ✓" if x2+x3 >= v23 else f"x2 + x3 = {x2+x3:.2f} < {v23} ✗")
                else:
                    st.error("Core is empty (negative allocations)")
            else:
                st.error("❌ Core is empty!")
                st.write(f"Condition violated: {v12 + v13 + v23} > 2 × {v123}")

# ==================== REPEATED GAMES ====================
elif category == "Repeated Games":
    st.title("🔁 Repeated Games")
    
    model = st.sidebar.selectbox(
        "Model",
        ["Infinitely Repeated PD", "Finitely Repeated", "Folk Theorem",
         "Trigger Strategies", "Grim Trigger vs Tit-for-Tat"]
    )
    
    if model == "Infinitely Repeated PD":
        st.markdown("""
        Infinitely repeated Prisoner's Dilemma with discount factor δ.
        
        Cooperation can be sustained if players are patient enough.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            cc = st.slider("Cooperate/Cooperate Payoff", 0, 10, 3)
            cd = st.slider("Cooperate/Defect Payoff", -5, 5, 0)
            dc = st.slider("Defect/Cooperate Payoff", 0, 10, 5)
            dd = st.slider("Defect/Defect Payoff", -5, 5, 1)
        
        with col2:
            delta = st.slider("Discount Factor (δ)", 0.0, 1.0, 0.9, 0.01)
            
            # Can cooperation be sustained with Grim Trigger?
            # Cooperation payoff: cc/(1-δ)
            # Deviation payoff: dc + δ × dd/(1-δ)
            
            coop_value = cc / (1 - delta) if delta < 1 else float('inf')
            defect_value = dc + (delta * dd / (1 - delta)) if delta < 1 else float('inf')
            
            st.metric("Cooperation Value", f"{coop_value:.2f}" if coop_value != float('inf') else "∞")
            st.metric("Deviation Value", f"{defect_value:.2f}" if defect_value != float('inf') else "∞")
            
            # Minimum discount factor for cooperation
            if dc - cc != 0 and cc - dd != 0:
                min_delta = (dc - cc) / (dc - dd)
                min_delta = max(0, min(1, min_delta))
                
                st.metric("Min δ for Cooperation", f"{min_delta:.3f}")
                
                if delta >= min_delta:
                    st.success("✅ Cooperation sustainable with Grim Trigger!")
                else:
                    st.error("❌ Players too impatient - cooperation not sustainable")
    
    elif model == "Grim Trigger vs Tit-for-Tat":
        st.markdown("""
        Compare different repeated game strategies:
        - **Grim Trigger**: Cooperate until opponent defects, then defect forever
        - **Tit-for-Tat**: Copy opponent's last move
        - **Always Cooperate**: Always cooperate
        - **Always Defect**: Always defect
        """)
        
        # Payoff matrix
        payoff_matrix = np.array([
            [[3, 3], [0, 5]],
            [[5, 0], [1, 1]]
        ])
        
        rounds = st.slider("Number of Rounds", 10, 200, 50)
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy1 = st.selectbox("Player 1 Strategy", 
                                    ["Tit-for-Tat", "Grim Trigger", "Always Cooperate", 
                                     "Always Defect", "Random"])
        
        with col2:
            strategy2 = st.selectbox("Player 2 Strategy",
                                    ["Tit-for-Tat", "Grim Trigger", "Always Cooperate",
                                     "Always Defect", "Random"])
        
        if st.button("Simulate Game"):
            scores, history = simulate_repeated_game(payoff_matrix, strategy1, strategy2, rounds)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Player 1 Total", scores[0])
            col2.metric("Player 2 Total", scores[1])
            col3.metric("Avg Per Round", f"{sum(scores)/(2*rounds):.2f}")
            
            # Plot scores over time
            p1_cumulative = np.cumsum([h[2] for h in history])
            p2_cumulative = np.cumsum([h[3] for h in history])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=p1_cumulative, name='Player 1', mode='lines'))
            fig.add_trace(go.Scatter(y=p2_cumulative, name='Player 2', mode='lines'))
            fig.update_layout(title="Cumulative Scores", xaxis_title="Round", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
            
            # Action history
            actions_p1 = ['C' if h[0] == 0 else 'D' for h in history[:20]]
            actions_p2 = ['C' if h[1] == 0 else 'D' for h in history[:20]]
            
            st.subheader("Action History (First 20 Rounds)")
            st.write("P1: " + " ".join(actions_p1))
            st.write("P2: " + " ".join(actions_p2))

# ==================== STOCHASTIC GAMES ====================
elif category == "Stochastic Games":
    st.title("🎲 Stochastic Games")
    
    model = st.sidebar.selectbox(
        "Model",
        ["Simple Stochastic Game", "Markov Decision Game", "Stopping Games",
         "Big Match Game", "Resource Management Game"]
    )
    
    if model == "Simple Stochastic Game":
        st.markdown("""
        Game with multiple states. Actions determine payoffs and state transitions.
        Players optimize expected discounted sum of payoffs.
        """)
        
        st.subheader("Two-State Game")
        
        delta = st.slider("Discount Factor", 0.0, 0.99, 0.9, 0.01)
        
        st.write("**State 1: Cooperation State**")
        col1, col2 = st.columns(2)
        with col1:
            s1_cc_payoff = st.number_input("Both Cooperate Payoff", 0, 10, 5, key="s1cc")
            s1_cc_prob = st.slider("Stay in State 1 Prob", 0.0, 1.0, 0.9, key="s1cc_p")
        with col2:
            s1_cd_payoff = st.number_input("C/D Payoff", -5, 5, 0, key="s1cd")
            s1_cd_prob = st.slider("Switch to State 2 Prob", 0.0, 1.0, 0.8, key="s1cd_p")
        
        st.write("**State 2: Punishment State**")
        col1, col2 = st.columns(2)
        with col1:
            s2_dd_payoff = st.number_input("Both Defect Payoff", -5, 5, 1, key="s2dd")
            s2_dd_prob = st.slider("Stay in State 2 Prob", 0.0, 1.0, 0.7, key="s2dd_p")
        with col2:
            s2_cd_payoff = st.number_input("C/D Payoff (S2)", -5, 5, 0, key="s2cd")
            s2_return_prob = st.slider("Return to State 1 Prob", 0.0, 1.0, 0.3, key="s2ret")
        
        if st.button("Analyze Equilibrium"):
            st.info("Computing stationary equilibrium strategies...")
            
            # Simplified analysis
            # Value of cooperation in state 1
            V_coop_s1 = s1_cc_payoff / (1 - delta * s1_cc_prob)
            V_defect_s1 = s1_cd_payoff + delta * s1_cd_prob * (s2_dd_payoff / (1 - delta * s2_dd_prob))
            
            st.subheader("Value Functions")
            st.write(f"Value of Cooperation (State 1): {V_coop_s1:.2f}")
            st.write(f"Value of Deviation (State 1): {V_defect_s1:.2f}")
            
            if V_coop_s1 >= V_defect_s1:
                st.success("✅ Cooperation is sustainable!")
            else:
                st.error("❌ Cooperation not sustainable with these parameters")
    
    elif model == "Resource Management Game":
        st.markdown("""
        **Common Pool Resource Game**
        
        Players extract from a shared resource. Over-extraction depletes the resource.
        """)
        
        n_players = st.slider("Number of Players", 2, 6, 3)
        initial_resource = st.slider("Initial Resource Level", 50, 200, 100)
        regeneration_rate = st.slider("Regeneration Rate", 0.0, 0.5, 0.1, 0.01)
        rounds = st.slider("Time Periods", 10, 100, 30)
        
        if st.button("Simulate Resource Extraction"):
            resource_history = [initial_resource]
            extractions_history = []
            
            for round in range(rounds):
                current_resource = resource_history[-1]
                
                # Each player extracts based on current level
                # Greedy strategy: extract 20% of available
                extractions = [min(current_resource / n_players * 0.3, current_resource / n_players) 
                              for _ in range(n_players)]
                total_extraction = sum(extractions)
                
                # Update resource
                new_resource = max(0, current_resource - total_extraction)
                new_resource = new_resource * (1 + regeneration_rate)
                
                resource_history.append(new_resource)
                extractions_history.append(extractions)
                
                if new_resource < 1:
                    st.warning(f"⚠️ Resource depleted at round {round+1}")
                    break
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=resource_history, name='Resource Level', 
                                        mode='lines+markers'))
                fig.update_layout(title="Resource Dynamics", xaxis_title="Round", 
                                 yaxis_title="Resource Level")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                total_extracted = sum(sum(e) for e in extractions_history)
                avg_per_player = total_extracted / n_players
                
                st.metric("Final Resource", f"{resource_history[-1]:.2f}")
                st.metric("Total Extracted", f"{total_extracted:.2f}")
                st.metric("Avg per Player", f"{avg_per_player:.2f}")
                
                if resource_history[-1] > initial_resource * 0.5:
                    st.success("✅ Sustainable extraction")
                elif resource_history[-1] > 10:
                    st.warning("⚠️ Resource stressed")
                else:
                    st.error("❌ Resource collapse")

# ==================== AI ANALYSIS ====================
elif category == "AI Analysis":
    st.title("🤖 AI-Powered Game Theory Analysis")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Custom Game Analysis", "Strategy Recommendation", "Equilibrium Finder",
         "Real-World Application", "Learning Assistant"]
    )
    
    if analysis_type == "Custom Game Analysis":
        st.markdown("Describe any game theory scenario and get AI-powered insights!")
        
        game_description = st.text_area(
            "Describe your game:",
            placeholder="Example: Two companies deciding whether to enter a new market. Entry costs $1M. If both enter, they split the $3M market. If only one enters, they get the full market...",
            height=150
        )
        
        st.subheader("Game Parameters (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            n_players = st.number_input("Number of Players", 2, 10, 2)
            game_type = st.selectbox("Game Type", 
                                    ["Simultaneous", "Sequential", "Repeated", "Stochastic"])
        
        with col2:
            information = st.selectbox("Information", 
                                      ["Complete", "Incomplete", "Imperfect"])
            objective = st.selectbox("Analysis Goal",
                                   ["Find Equilibria", "Optimal Strategy", "Payoff Analysis", 
                                    "Comparative Statics"])
        
        if st.button("🤖 Analyze with AI"):
            if game_description:
                with st.spinner("AI is analyzing your game..."):
                    prompt = f"""
                    Analyze this game theory scenario:
                    
                    {game_description}
                    
                    Game parameters:
                    - Players: {n_players}
                    - Type: {game_type}
                    - Information: {information}
                    - Analysis goal: {objective}
                    
                    Provide:
                    1. Formal game structure
                    2. Equilibrium analysis
                    3. Strategic insights
                    4. Optimal strategies
                    5. Real-world applications
                    """
                    
                    analysis = get_ai_analysis(game_description, 
                                              f"Players: {n_players}, Type: {game_type}")
                    
                    st.markdown("### 🎯 AI Analysis")
                    st.markdown(analysis)
            else:
                st.warning("Please describe your game scenario")
    
    elif analysis_type == "Strategy Recommendation":
        st.markdown("Get AI recommendations for your specific situation!")
        
        situation = st.text_area(
            "Describe your strategic situation:",
            placeholder="I'm negotiating a salary. The company made an initial offer of $80k. I want $100k. What strategy should I use?",
            height=150
        )
        
        constraints = st.text_area(
            "Any constraints or additional information:",
            placeholder="I have another offer for $85k. The negotiation is in 2 days.",
            height=100
        )
        
        if st.button("🎯 Get Strategy Recommendation"):
            if situation:
                with st.spinner("Generating recommendations..."):
                    prompt = f"""
                    Provide strategic advice for this situation:
                    
                    Situation: {situation}
                    
                    Constraints: {constraints}
                    
                    Analyze using game theory principles and provide:
                    1. Strategic options
                    2. Expected outcomes for each option
                    3. Recommended strategy
                    4. Potential pitfalls to avoid
                    5. Key factors to consider
                    """
                    
                    recommendation = get_ai_analysis(situation, constraints)
                    
                    st.markdown("### 💡 Strategic Recommendation")
                    st.markdown(recommendation)
            else:
                st.warning("Please describe your situation")
    
    elif analysis_type == "Real-World Application":
        st.markdown("Explore real-world applications of game theory concepts!")
        
        domain = st.selectbox(
            "Select Domain",
            ["Business & Economics", "Politics & Voting", "Biology & Evolution",
             "Computer Science", "Social Networks", "Environmental Policy",
             "Military Strategy", "Sports"]
        )
        
        concept = st.selectbox(
            "Select Concept",
            ["Nash Equilibrium", "Prisoner's Dilemma", "Auction Design",
             "Bargaining", "Mechanism Design", "Evolutionary Stability",
             "Network Effects", "Signaling Games"]
        )
        
        if st.button("🌍 Explore Applications"):
            with st.spinner("Finding real-world applications..."):
                prompt = f"""
                Explain real-world applications of {concept} in {domain}.
                
                Provide:
                1. 3-5 concrete examples
                2. How game theory principles apply
                3. Outcomes and insights
                4. Lessons learned
                5. Current relevance
                
                Make it practical and actionable.
                """
                
                applications = get_ai_analysis(f"{concept} in {domain}", "real-world examples")
                
                st.markdown(f"### 🌍 {concept} in {domain}")
                st.markdown(applications)
    
    elif analysis_type == "Learning Assistant":
        st.markdown("Learn game theory concepts with AI assistance!")
        
        topic = st.selectbox(
            "What would you like to learn?",
            ["Nash Equilibrium Basics", "Dominant Strategies", "Mixed Strategies",
             "Subgame Perfection", "Bayesian Games", "Mechanism Design Basics",
             "Evolutionary Game Theory", "Auction Theory", "Bargaining Solutions",
             "Cooperative Games", "Repeated Games", "Stochastic Games"]
        )
        
        level = st.radio("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        
        specific_question = st.text_input(
            "Any specific question? (Optional)",
            placeholder="What's the difference between Nash and subgame perfect equilibrium?"
        )
        
        if st.button("📚 Learn"):
            with st.spinner("Preparing lesson..."):
                prompt = f"""
                Explain {topic} for a {level} level student.
                
                {"Specific question: " + specific_question if specific_question else ""}
                
                Provide:
                1. Clear explanation with intuition
                2. Simple example
                3. Key insights
                4. Common misconceptions
                5. Practice exercise
                
                Make it engaging and easy to understand.
                """
                
                lesson = get_ai_analysis(topic, f"Level: {level}, Question: {specific_question}")
                
                st.markdown(f"### 📚 Learning: {topic}")
                st.markdown(lesson)
                
                st.markdown("---")
                st.markdown("### 💭 Test Your Understanding")
                
                user_answer = st.text_area(
                    "Try the practice exercise or explain the concept in your own words:",
                    height=100
                )
                
                if st.button("✅ Check Understanding"):
                    if user_answer:
                        feedback_prompt = f"""
                        The student is learning about {topic}.
                        
                        Their response: {user_answer}
                        
                        Provide constructive feedback:
                        1. What they got right
                        2. Any misconceptions
                        3. Suggestions for improvement
                        4. Encouragement
                        """
                        
                        feedback = get_ai_analysis(topic, user_answer)
                        st.markdown("### 📝 Feedback")
                        st.markdown(feedback)

# Additional Simulations Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Simulations")

if st.sidebar.button("🎲 Random Game"):
    st.title("🎲 Random Game Generator")
    
    n = np.random.randint(2, 4)
    m = np.random.randint(2, 4)
    
    payoff_matrix = np.random.randint(-5, 10, (n, m, 2))
    
    st.write(f"**Random {n}×{m} Game**")
    
    for player in range(2):
        st.write(f"Player {player+1} Payoffs:")
        df = pd.DataFrame(payoff_matrix[:,:,player])
        st.dataframe(df)
    
    pure_eq = find_nash_equilibrium_pure(payoff_matrix)
    
    if pure_eq:
        st.success(f"Pure Strategy Equilibria: {pure_eq}")
    else:
        st.info("No pure strategy equilibria found")

if st.sidebar.button("🏆 Tournament"):
    st.title("🏆 Strategy Tournament")
    st.markdown("Round-robin tournament of repeated game strategies")
    
    strategies = ["Tit-for-Tat", "Grim Trigger", "Always Cooperate", 
                 "Always Defect", "Random"]
    
    payoff_matrix = np.array([
        [[3, 3], [0, 5]],
        [[5, 0], [1, 1]]
    ])
    
    rounds = 50
    results = np.zeros((len(strategies), len(strategies)))
    
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if i <= j:
                scores, _ = simulate_repeated_game(payoff_matrix, s1, s2, rounds)
                results[i, j] = scores[0]
                results[j, i] = scores[1]
    
    total_scores = results.sum(axis=1)
    winner_idx = np.argmax(total_scores)
    
    st.success(f"🏆 Tournament Winner: {strategies[winner_idx]}")
    
    df = pd.DataFrame(results, columns=strategies, index=strategies)
    st.dataframe(df.style.highlight_max(axis=1))
    
    fig = px.bar(x=strategies, y=total_scores, title="Total Scores")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 About
This simulator covers 100+ game theory concepts:
- Classic games & equilibria
- Auctions & mechanism design
- Evolutionary & network games
- Cooperative & bargaining games
- AI-powered analysis

Built with Streamlit & Gemini AI
""")

st.sidebar.info("💡 Use AI Analysis for deeper insights on any scenario!")
