import numpy as np
import random
import matplotlib.pyplot as plt


ALLOCATION_PER_STEP = 10
BETA = 0.5  # Weight for averaging past and current rates
EXPERIMENTS=50
class UE:
    def __init__(self, ue_id, traffic_type):
        self.ue_id = ue_id
        self.traffic_type = traffic_type
        self.channel_quality = self.calculate_initial_channel_quality()
        self.traffic_demand = self.generate_traffic_demand(traffic_type)
        self.allocated_bandwidth = 0
        self.remaining_demand = self.traffic_demand
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0

    def calculate_initial_channel_quality(self):
        return random.uniform(0.2, 1.0)  # Random channel quality between 0.2 and 1.0

    def update_channel_quality(self):
        self.channel_quality = max(0.2, min(1.0, self.channel_quality + random.uniform(-0.3, 0.3)))

    def generate_traffic_demand(self, traffic_type):
        min_demand, max_demand = TRAFFIC_TYPES[traffic_type]
        return random.uniform(min_demand, max_demand)

    def allocate_bandwidth(self, amount):
        allocation = min(amount, self.remaining_demand)
        self.allocated_bandwidth += allocation
        self.remaining_demand -= allocation
        if allocation > 0:
            self.total_latency += 1 - (allocation / self.traffic_demand)
        self.instantaneous_rate = allocation
        self.update_average_rate()

    def update_average_rate(self):
        self.average_rate = (1 - BETA) * self.average_rate + BETA * self.instantaneous_rate

    def reset(self):
        self.allocated_bandwidth = 0
        self.remaining_demand = self.traffic_demand
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0

def create_users(num_users):
    users = []
    for i in range(num_users):
        traffic_type = random.choice(list(TRAFFIC_TYPES.keys()))
        users.append(UE(i, traffic_type))
    return users

def sequential_round_robin(users, total_bandwidth, allocation_per_step):
    experiment = 1

    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    while experiment <= EXPERIMENTS:
        # Reset user demands and metrics at the start of each time step
        for user in users:
            user.reset()

        step_count = 0  # Reset step count for the current time step
        num_requests = random.randint(int(NUM_USERS / 2), int(NUM_USERS))  # Number of active users
        requesting_users = random.sample(users, num_requests)  # Select random users for this time step
        requesting_users.sort(key=lambda u: u.ue_id)  # Ensure round-robin order by user ID

        print(f"\n--- Experiment {experiment} ---")
        print(f"Users requesting bandwidth: {[user.ue_id for user in requesting_users]}")

        remaining_bandwidth = total_bandwidth  # Reset total bandwidth for this time step
        i = 0  # Index to cycle through requesting users in round-robin fashion

        # Bandwidth allocation loop
        while remaining_bandwidth > 0 and any(user.remaining_demand > 0 for user in requesting_users):
            user = requesting_users[i % len(requesting_users)]  # Cycle through users in round-robin order

            if user.remaining_demand > 0:
                # Calculate the bandwidth sent and effective allocation
                bandwidth_sent = min(
                    allocation_per_step,
                    remaining_bandwidth,
                    user.remaining_demand / user.channel_quality
                )
                effective_allocation = bandwidth_sent * user.channel_quality
                remaining_bandwidth -= bandwidth_sent  # Update remaining bandwidth
                user.allocate_bandwidth(effective_allocation)  # Allocate bandwidth to the user
                step_count += 1  # Increment step count for each valid allocation

                # Debugging output
                print(f"\nStep {step_count}: Allocated {effective_allocation:.2f} Mbps to UE {user.ue_id}")
                print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
                print(f"Total Bandwidth Remaining = {remaining_bandwidth:.2f} Mbps")
                print(f"UE {user.ue_id} - Bandwidth Sent by BS: {bandwidth_sent:.2f} Mbps, "
                      f"Effective Allocation: {effective_allocation:.2f} Mbps, "
                      f"Channel Quality: {user.channel_quality:.2f}")

            i += 1  # Move to the next user in round-robin order

        # Collect metrics from the requesting users
        fairness_over_time.append(calculate_fairness(requesting_users))  # Calculate fairness for this time step
        throughput_over_time.append(calculate_total_throughput(requesting_users))  # Calculate throughput
        latency_over_time.append(calculate_average_latency(requesting_users))  # Calculate average latency

        # Debugging output for metrics
        print("\nMetrics for Time Step:")
        print(f"Fairness Index: {fairness_over_time[-1]:.2f}")
        print(f"Total Throughput: {throughput_over_time[-1]:.2f} Mbps")
        print(f"Average Latency: {latency_over_time[-1]:.2f} time steps")

        # Move to the next time step
        experiment += 1

    # Final metrics output
    print("\nFinal Metrics After All Time Steps:")
    print(f"Fairness Over Time: {fairness_over_time}")
    print(f"Throughput Over Time: {throughput_over_time}")
    print(f"Latency Over Time: {latency_over_time}")

    return fairness_over_time, throughput_over_time, latency_over_time


def proportional_fair_scheduler(users, total_bandwidth, allocation_per_step):
    experiment = 1

    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    while experiment <= EXPERIMENTS:
        # Reset remaining demand for each user at the start of each time step
        for user in users:
            user.reset()

        step_count = 0  # Reset step count for this time step

        # Update channel qualities for all users
        for user in users:
            user.update_channel_quality()

        # Randomly select requesting users
        num_requests = random.randint(int(NUM_USERS / 2), int(NUM_USERS))
        requesting_users = random.sample(users, num_requests)

        print(f"\n--- Experiment {experiment} ---")
        print(f"Users requesting bandwidth: {[user.ue_id for user in requesting_users]}")

        # Calculate PF metrics
        metrics = [
            (user, (user.channel_quality * allocation_per_step) / max(user.average_rate, 0.1))
            for user in requesting_users
        ]
        metrics.sort(key=lambda x: x[1], reverse=True)

        if not metrics:
            print("No users with remaining demand.")
            break

        # Bandwidth allocation loop
        remaining_experiment_bandwidth = total_bandwidth
        i = 0

        while remaining_experiment_bandwidth > 0 and any(user.remaining_demand > 0 for user, _ in metrics):
            user = metrics[i % len(metrics)][0]  # Cycle through sorted metrics

            if user.remaining_demand <= 0:
                i += 1  # Skip users with no demand
                continue

            # Calculate bandwidth allocation
            bandwidth_sent = min(
                allocation_per_step,
                remaining_experiment_bandwidth,
                user.remaining_demand / user.channel_quality
            )
            effective_allocation = bandwidth_sent * user.channel_quality
            remaining_experiment_bandwidth -= bandwidth_sent
            user.allocate_bandwidth(effective_allocation)
            step_count += 1

            print(f"\nStep {step_count}: Allocated {effective_allocation:.2f} Mbps to UE {user.ue_id}")
            print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
            print(f"Total Bandwidth Remaining = {remaining_experiment_bandwidth:.2f} Mbps")

            i += 1

        # Collect metrics
        users_in_metrics = [user for user, _ in metrics]
        fairness_over_time.append(calculate_fairness(users_in_metrics))
        throughput_over_time.append(calculate_total_throughput(users_in_metrics))
        latency_over_time.append(calculate_average_latency(users_in_metrics))

        experiment += 1

    return (
        fairness_over_time,
        throughput_over_time,
        latency_over_time
    )


    
def calculate_total_throughput(users):
    return sum(user.allocated_bandwidth for user in users)

def calculate_average_latency(users):
    """Calculate the average latency for all UEs."""
    latencies = [user.total_latency for user in users if user.total_latency > 0]
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)

def calculate_fairness(users):
    """Calculate Jain's fairness index considering traffic demands."""
    allocated_bandwidths = [user.allocated_bandwidth / user.traffic_demand for user in users if user.allocated_bandwidth > 0]
    if not allocated_bandwidths:
        return 0.0
    n = len(allocated_bandwidths)
    squared_sum = sum(allocated_bandwidths)**2
    sum_squared = sum(x*x for x in allocated_bandwidths)
    return squared_sum / (n * sum_squared) if sum_squared > 0 else 0.0

def print_user_status(users, title):
    print(f"\n--- {title} ---")
    print("UE | Type            | Allocated  | Remaining Demand | Channel Quality")
    for user in users:
        print(
            f"{str(user.ue_id).ljust(2)} | {user.traffic_type.ljust(15)} | "
            f"{f'{user.allocated_bandwidth:.2f}'.rjust(10)} Mbps | "
            f"{f'{user.remaining_demand:.2f}'.rjust(15)} Mbps | "
            f"{user.channel_quality:.3f}"
        )
    print("\n")

def plot_metrics(metrics, labels, title, ylabel, save_as=None):
    plt.figure(figsize=(10, 6))
    for metric, label in zip(metrics, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
    plt.show()

def run_simulation():
    global NUM_USERS, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP, TRAFFIC_TYPES

    # Scenario selection 
    print("\nSelect test scenario:")
    print("1. 200 users - 300 Mbps")
    print("2. 150 users - 200 Mbps")
    print("3. 100 users - 100 Mbps")
    
    scenario = input("Enter scenario number (default=1): ") or "1"
    
    if scenario == "1":
        # High Mobility Scenario
        NUM_USERS = 200        
        TOTAL_BANDWIDTH = 300  
        TRAFFIC_TYPES = {
                "video_streaming": (4, 10),  # HD streaming
                "web_browsing": (1, 2),    # Light web activity
                "voice_call": (0.1, 0.5)   # VoIP
            }
            
        print("\nRunning Scenario 1:")
        print("- 200 users")
        print("- Bandwidth (300 Mbps)")

    elif scenario == "2":
        # Mixed Traffic Scenario
        NUM_USERS = 150            
        TOTAL_BANDWIDTH = 200 
        TRAFFIC_TYPES = {
                "video_streaming": (5, 12),  # HD and some Ultra HD
                "web_browsing": (1, 4),     # Moderate browsing
                "voice_call": (0.3, 1)      # VoIP and video calls
            }

        print("\nRunning Scenario 2:")
        print("- 150 users")
        print("- Bandwidth (200 Mbps)")

    elif scenario == "3":
        # Cell Edge Scenario
        NUM_USERS = 100           
        TOTAL_BANDWIDTH = 100
        TRAFFIC_TYPES = {
                "video_streaming": (3, 9),  # SD/low-HD streaming
                "web_browsing": (0.5, 2),  # Light browsing
                "voice_call": (0.1, 0.9)   # VoIP
            }
        print("\nRunning Scenario 3:")
        print("- 100 users")
        print("- Bandwidth (100 Mbps)")

    users = create_users(NUM_USERS)

    # Print initial environment
    print("\nInitial Environment Setup:")
    print("UE | Type            | Demand     | Channel Quality")
    for user in users:
        print(f"{str(user.ue_id).ljust(2)} | {user.traffic_type.ljust(15)} | "
                f"{f'{user.traffic_demand:.2f}'.rjust(5)} Mbps | "
                f"{user.channel_quality:.2f}".ljust(14))

    # Run Round-Robin
    print("\n=== Running Sequential Round-Robin Scheduler ===")
    (
        fairness_over_time_rr,
        throughput_over_time_rr,
        latency_over_time_rr,
    ) = sequential_round_robin(users, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP)


    # Reset users for Proportional Fair
    for user in users:
        user.reset()

    # Run Proportional Fair
    print("\n=== Running Proportional Fair Scheduler ===")
    (
        fairness_over_time_pf,
        throughput_over_time_pf,
        latency_over_time_pf,
    ) = proportional_fair_scheduler(users, TOTAL_BANDWIDTH, ALLOCATION_PER_STEP)

    # Plot metrics
    plot_metrics(
        [fairness_over_time_rr, fairness_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Fairness Over Time - Scenario {scenario}",
        "Fairness Index",
        save_as=f"fairness_scenario_{scenario}.png"
    )

    plot_metrics(
        [throughput_over_time_rr, throughput_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Throughput Over Time - Scenario {scenario}",
        "Throughput (Mbps)",
        save_as=f"throughput_scenario_{scenario}.png"
    )

    plot_metrics(
        [latency_over_time_rr, latency_over_time_pf],
        ["Round-Robin", "Proportional Fair"],
        f"Average Latency Over Time - Scenario {scenario}",
        "Latency (time steps)",
        save_as=f"latency_scenario_{scenario}.png"
    )

if __name__ == "__main__":
    random.seed(3)
    run_simulation()