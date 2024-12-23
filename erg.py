import random
import matplotlib.pyplot as plt

BETA = 0.5  #Weight for averaging past and current rates
SIMULATION_DURATION = 100 #seconds
CONGESTION_PROBABILITY = 0.1

class UE:
    def __init__(self, ue_id, traffic_type):
        self.ue_id = ue_id
        self.traffic_type = traffic_type
        self.channel_quality = self.calculate_initial_channel_quality()
        self.traffic_demand = self.generate_traffic_demand(traffic_type)
        self.allocated_bandwidth = 0 #bandwidth allocated after the last time UE became active (seconds>0)
        self.seconds = 0
        self.remaining_demand = 0 #self.traffic_demand * self.seconds
        self.total_demand = 0 #self.traffic_demand * 'initial seconds'
        self.total_latency = 0
        self.instantaneous_rate = 0
        self.average_rate = 0

    def calculate_initial_channel_quality(self):
        return random.uniform(0.2, 1.0)  #Random channel quality between 0.2 and 1.0

    def update_channel_quality(self):
        self.channel_quality = max(0.9, min(1.0, self.channel_quality + random.uniform(-0.1, 0.1)))

    def generate_traffic_demand(self, traffic_type):
        min_demand, max_demand = TRAFFIC_TYPES[traffic_type]
        return random.uniform(min_demand, max_demand)

    def allocate_bandwidth(self):
        self.allocated_bandwidth += self.traffic_demand
        self.remaining_demand -= self.traffic_demand
        # # self.total_latency += 1 - (allocation / self.traffic_demand)
        self.instantaneous_rate = self.traffic_demand
        self.update_average_rate()
        self.seconds -= 1

    def update_average_rate(self):
        self.average_rate = (1 - BETA) * self.average_rate + BETA * self.instantaneous_rate

    def reset(self, congested=False):
        PROBABILITY_TO_ACTIVATE_USER = 0.1
        PROBABILITY_TO_ACTIVATE_USER_WHEN_CONGESTED = 0.8
        if self.seconds == 0:
            if congested:
                if random.random() < PROBABILITY_TO_ACTIVATE_USER_WHEN_CONGESTED:
                    self.seconds = random.randint(1, 30)
                    self.allocated_bandwidth = 0
                    self.remaining_demand = self.traffic_demand * self.seconds
                    self.total_demand = self.seconds * self.traffic_demand
            elif random.random() < PROBABILITY_TO_ACTIVATE_USER:
                self.seconds = random.randint(1, 30)
                self.allocated_bandwidth = 0
                self.remaining_demand = self.traffic_demand * self.seconds
                self.total_demand = self.seconds * self.traffic_demand
            self.total_latency = 0
            self.instantaneous_rate = 0
            self.average_rate = 0


def create_users(num_users):
    users = []
    for i in range(num_users):
        traffic_type = random.choice(list(TRAFFIC_TYPES.keys()))
        users.append(UE(i, traffic_type))
    return users

def sequential_round_robin(users, base_station_bandwidth):
    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    second_count = 0

    #Bandwidth allocation loop
    while second_count < SIMULATION_DURATION:
        remaining_bandwidth = base_station_bandwidth  #Reset total bandwidth for this second

        is_congested = random.random() < CONGESTION_PROBABILITY
        #Reset users (with seconds = 0)
        for user in users:
            user.total_latency += 1 #increase latency of all users by 1
            if is_congested:
                user.reset(True) #Happens if seconds == 0
            else:
                user.reset() #Happens if seconds == 0
            user.update_channel_quality()

        user_index = 0
        users = users[user_index:] + users[:user_index] 
        #Give bandwidth to users
        for user in users:
            if user.seconds > 0:
                #If you cant satisfy the demand of the user, skip the user
                if user.traffic_demand / user.channel_quality > remaining_bandwidth:
                    continue

                #Calculate the bandwidth sent and effective allocation
                bandwidth_sent = user.traffic_demand / user.channel_quality
                # effective_allocation = bandwidth_sent * user.channel_quality
                remaining_bandwidth -= bandwidth_sent  #Update remaining bandwidth
                # user.allocate_bandwidth(effective_allocation)  #Allocate bandwidth to the user
                user.allocate_bandwidth()  #Allocate bandwidth to the user

                # #Debugging output
                # print(f"\nUser Index {user_index}: Allocated {effective_allocation:.2f} Mbps to UE {user.ue_id}")
                # print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
                # print(f"Total Bandwidth Remaining = {remaining_bandwidth:.2f} Mbps")
                # print(f"UE {user.ue_id} - Bandwidth Sent by BS: {bandwidth_sent:.2f} Mbps, "
                #         f"Effective Allocation: {effective_allocation:.2f} Mbps, "
                #         f"Channel Quality: {user.channel_quality:.2f}")

                #Collect metrics from the requesting users
                fairness_over_time.append(calculate_fairness(users))  #Calculate fairness for this time step
                throughput_over_time.append(calculate_total_throughput(users))  #Calculate throughput
                latency_over_time.append(calculate_average_latency(users))  #Calculate average latency

                # #Debugging output for metrics
                # print("\nMetrics for Time Step:")
                # print(f"Fairness Index: {fairness_over_time[-1]:.2f}")
                # print(f"Total Throughput: {throughput_over_time[-1]:.2f} Mbps")
                # print(f"Average Latency: {latency_over_time[-1]:.2f} time steps")
            user_index += 1
            break
        second_count += 1
    # #Final metrics output
    # print("\nFinal Metrics After All Time Steps:")
    # print(f"Fairness Over Time: {fairness_over_time}")
    # print(f"Throughput Over Time: {throughput_over_time}")
    # print(f"Latency Over Time: {latency_over_time}")

    return fairness_over_time, throughput_over_time, latency_over_time


def proportional_fair_scheduler(users, base_station_bandwidth):
    # Metric trackers
    fairness_over_time = []
    throughput_over_time = []
    latency_over_time = []

    second_count = 0

    #Bandwidth allocation loop
    while second_count < SIMULATION_DURATION:
        remaining_bandwidth = base_station_bandwidth  #Reset total bandwidth for this second

        #Reset users (with seconds = 0)
        for user in users:
            user.total_latency += 1 #increase latency of all users by 1
            user.reset() #Happens if seconds == 0
            user.update_channel_quality()

        # Recalculate PF metrics for all users
        metrics = [
            (user, (user.channel_quality * base_station_bandwidth) / max(user.average_rate, 0.1))
            for user in users if user.seconds > 0
        ]
        # Sort users by PF metric (highest first)
        metrics.sort(key=lambda x: x[1], reverse=True)

        # Allocate bandwidth to the user with the highest PF metric
        # user = metrics[0][0]  # Select the user with the highest PF metric

        #Give bandwidth to users
        for [user, _] in metrics:
            if user.seconds > 0:
                #If you cant satisfy the demand of the user, skip the user
                if user.traffic_demand / user.channel_quality > remaining_bandwidth:
                    continue

                #Calculate the bandwidth sent and effective allocation
                bandwidth_sent = user.traffic_demand / user.channel_quality
                # effective_allocation = bandwidth_sent * user.channel_quality
                remaining_bandwidth -= bandwidth_sent  #Update remaining bandwidth
                # user.allocate_bandwidth(effective_allocation)  #Allocate bandwidth to the user
                user.allocate_bandwidth()  #Allocate bandwidth to the user

                # #Debugging output
                # print(f"\nUser Index {user_index}: Allocated {effective_allocation:.2f} Mbps to UE {user.ue_id}")
                # print(f"UE {user.ue_id}: Remaining Demand = {user.remaining_demand:.2f} Mbps")
                # print(f"Total Bandwidth Remaining = {remaining_bandwidth:.2f} Mbps")
                # print(f"UE {user.ue_id} - Bandwidth Sent by BS: {bandwidth_sent:.2f} Mbps, "
                #         f"Effective Allocation: {effective_allocation:.2f} Mbps, "
                #         f"Channel Quality: {user.channel_quality:.2f}")

                #Collect metrics from the requesting users
                fairness_over_time.append(calculate_fairness(users))  #Calculate fairness for this time step
                throughput_over_time.append(calculate_total_throughput(users))  #Calculate throughput
                latency_over_time.append(calculate_average_latency(users))  #Calculate average latency

                # #Debugging output for metrics
                # print("\nMetrics for Time Step:")
                # print(f"Fairness Index: {fairness_over_time[-1]:.2f}")
                # print(f"Total Throughput: {throughput_over_time[-1]:.2f} Mbps")
                # print(f"Average Latency: {latency_over_time[-1]:.2f} time steps")
            break
        second_count += 1
    # #Final metrics output
    # print("\nFinal Metrics After All Time Steps:")
    # print(f"Fairness Over Time: {fairness_over_time}")
    # print(f"Throughput Over Time: {throughput_over_time}")
    # print(f"Latency Over Time: {latency_over_time}")

    return fairness_over_time, throughput_over_time, latency_over_time


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
    allocated_bandwidths = [user.allocated_bandwidth / user.total_demand for user in users if user.seconds > 0]
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

def copy_user_list(list_to_copy):
    new_list = []
    for i in range(len(list_to_copy)):
        new_list.append(UE(i, list_to_copy[i].traffic_type))
    return new_list

def run_simulation():
    global NUM_USERS, BASE_STATION_BANDWIDTH, TRAFFIC_TYPES

    # Scenario selection 
    print("\nSelect test scenario:")
    print("1. 200 users - 300 Mbps")
    print("2. 150 users - 200 Mbps")
    print("3. 100 users - 100 Mbps")
    
    # scenario = input("Enter scenario number (default=1): ") or "1"
    scenario = "1"
    
    if scenario == "1":
        NUM_USERS = 100
        BASE_STATION_BANDWIDTH = 100 #Must be greater than the bandwidth need of any UE
        TRAFFIC_TYPES = {
                "video_streaming": (5, 10),  #HD streaming
                # "web_browsing": (1, 3),    #Light web activity
                # "voice_call": (0.2, 0.8)   #VoIP
            }
            
        print("\nRunning Scenario 1:")
        print("- 200 users")
        print("- Bandwidth (300 Mbps)")

    # elif scenario == "2":
       
    #     NUM_USERS = 150            
    #     BASE_STATION_BANDWIDTH = 200 
    #     TRAFFIC_TYPES = {
    #             "video_streaming": (5, 12),  #HD and some Ultra HD
    #             "web_browsing": (1, 4),     #Moderate browsing
    #             "voice_call": (0.3, 1)      #VoIP and video calls
    #         }

    #     print("\nRunning Scenario 2:")
    #     print("- 150 users")
    #     print("- Bandwidth (200 Mbps)")

    # elif scenario == "3":
    #     NUM_USERS = 100
    #     BASE_STATION_BANDWIDTH = 100
    #     TRAFFIC_TYPES = {
    #             "video_streaming": (3, 9),  #SD/low-HD streaming
    #             "web_browsing": (0.5, 2),  #Light browsing
    #             "voice_call": (0.1, 0.9)   #VoIP
    #         }
    #     print("\nRunning Scenario 3:")
    #     print("- 100 users")
    #     print("- Bandwidth (100 Mbps)")

    users = create_users(NUM_USERS)
    users_rr = copy_user_list(users)
    users_pf = copy_user_list(users)

    #Print initial environment
    print("\nInitial Environment Setup:")
    print("UE | Type            | Demand     | Channel Quality")
    for user in users:
        print(f"{str(user.ue_id).ljust(2)} | {user.traffic_type.ljust(15)} | "
                f"{f'{user.traffic_demand:.2f}'.rjust(5)} Mbps | "
                f"{user.channel_quality:.2f}".ljust(14))

    #Run Round-Robin
    print("\n=== Running Sequential Round-Robin Scheduler ===")
    (
        fairness_over_time_rr,
        throughput_over_time_rr,
        latency_over_time_rr,
    ) = sequential_round_robin(users_rr, BASE_STATION_BANDWIDTH)

    #Run Proportional Fair
    print("\n=== Running Proportional Fair Scheduler ===")
    (
        fairness_over_time_pf,
        throughput_over_time_pf,
        latency_over_time_pf,
    ) = proportional_fair_scheduler(users, BASE_STATION_BANDWIDTH)

    print(sum(fairness_over_time_rr) / len(fairness_over_time_rr))
    print(sum(fairness_over_time_pf) / len(fairness_over_time_pf))
    print(sum(throughput_over_time_rr) / len(throughput_over_time_rr))
    print(sum(throughput_over_time_pf) / len(throughput_over_time_pf))
    print(sum(latency_over_time_rr) / len(latency_over_time_rr))
    print(sum(latency_over_time_pf) / len(latency_over_time_pf))

    # #Plot metrics
    plot_metrics(
        [fairness_over_time_rr, fairness_over_time_pf],
        # [fairness_over_time_rr, fairness_over_time_rr],
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
    import time
    seed = time.time()
    print("seed: ", seed)
    random.seed(seed)
    run_simulation()