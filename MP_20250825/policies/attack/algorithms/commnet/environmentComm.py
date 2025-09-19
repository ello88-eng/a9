import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import io

import PIL.Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

os.chdir(os.path.dirname(__file__))

EP_LEN = 60
N_SITE = 5
N_BOMB = 200
N_COMMAGENT = 10
N_DQNAGENT = 0
GRID = 32000  # [m]
OBSERVABLE = GRID * 3
H_MAX = 600  # [m]
H_MIN = 0  # [m]
VELOCITY = 4425  # [m/min]
Max_num_bomb = 4

font = {"size": 10}
plt.rc("font", **font)  # pass in the font dict as kwargs
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})


class Site:

    def __init__(self, id=None, x=None, y=None, z=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


class Bomb:

    def __init__(self, id=None, Departure=None, Arrival=None):
        self.id = id
        self.Arrival = Arrival
        self.Departure = Departure
        self.connect = {"drone": -1, "isOnboard": 0, "isArrive": 0}

    def is_support(self):
        return (
            self.connect["drone"],
            self.connect["isOnboard"],
            self.connect["isArrive"],
        )


class Drone:

    def __init__(self, id=None, x=None, y=None, z=None, site=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.site = site
        self.onboarding = 0
        self.counter = 4
        self.takeoff = 0
        self.Max_num_bomb = Max_num_bomb
        self.numBomb = 0
        self.distance = 0
        self.isworking = 1
        self.seat = np.full([self.Max_num_bomb], -1, dtype=int)
        self.batteryRemain = 4 * 5.4 * 100000000
        self.maxbattery = self.batteryRemain
        self.num = 0
        self.numidx = np.zeros([N_SITE])
        self.numidx[site] = 1

    def batteryConsump(self, action):
        if (
            action == 4
            or action == 6
            or action == 7
            or action == 8
            or action == 9
            or action == 10
        ):
            batteryconsumption = 622 * 1000 * 60  # [J]
        elif action == 5:
            batteryconsumption = -self.maxbattery * 0.2 / 5
        else:
            batteryconsumption = 230 * 1000 * 60  # [J]
        self.batteryRemain -= batteryconsumption
        self.batteryRemain = min(self.batteryRemain, self.maxbattery)
        self.batteryRemain = max(self.batteryRemain, 0)
        if self.batteryRemain == 0:
            self.isworking = 0

    def _avail_action(self, bombList, siteList):
        avail_action = np.ones(15)
        if self.x >= GRID or self.z == H_MIN:
            avail_action[0] = 0
        if self.x <= -GRID or self.z == H_MIN:
            avail_action[1] = 0
        if self.y >= GRID or self.z == H_MIN:
            avail_action[2] = 0
        if self.y <= -GRID or self.z == H_MIN:
            avail_action[3] = 0
        if (
            self.x >= (GRID / math.sqrt(2))
            or self.y >= (GRID / math.sqrt(2))
            or self.z == H_MIN
        ):
            avail_action[4] = 0
        if (
            self.x <= (-GRID / math.sqrt(2))
            or self.y >= (GRID / math.sqrt(2))
            or self.z == H_MIN
        ):
            avail_action[5] = 0
        if (
            self.x >= (GRID / math.sqrt(2))
            or self.y <= (-GRID / math.sqrt(2))
            or self.z == H_MIN
        ):
            avail_action[6] = 0
        if (
            self.x <= (-GRID / math.sqrt(2))
            or self.y <= (-GRID / math.sqrt(2))
            or self.z == H_MIN
        ):
            avail_action[7] = 0
        avail_action[8:] = 0  # 8 ~ 14
        if self.z == H_MAX:
            for v in range(N_SITE):  # 10 11 12 13 14
                diff = math.sqrt(
                    (siteList[v].x - self.x) ** 2 + (siteList[v].y - self.y) ** 2
                )
            if self.site != -1:
                avail_action[self.site + 10] = 0
        if self.z == H_MIN:
            if self.counter == 5:
                avail_action[8] = 1
                avail_action[9] = 0
            elif self.counter < 5:
                avail_action = np.zeros(15)
                avail_action[9] = 1
        return avail_action

    def arrive_process(self, bombList, idx, id, siteList):
        bombList[id].connect["isOnboard"] = 0
        bombList[id].connect["isArrive"] = 1
        self.seat[idx] = -1
        self.takeoff += 1
        x_diff = siteList[bombList[id].Departure].x - siteList[bombList[id].Arrival].x
        y_diff = siteList[bombList[id].Departure].y - siteList[bombList[id].Arrival].y
        self.distance = math.sqrt((x_diff) ** 2 + (y_diff) ** 2)

    def onboard_process(self, bomb, idx):
        bomb.connect["drone"] = self.id
        bomb.connect["isOnboard"] = 1
        self.seat[idx] = bomb.id

    def transition(self, action, bombList, siteList):
        if self.isworking == 1:
            if action == 0:
                self.x += VELOCITY  # Down
            elif action == 1:
                self.x -= VELOCITY  # Up
            elif action == 2:
                self.y += VELOCITY  # Right
            elif action == 3:
                self.y -= VELOCITY  # Left
            elif action == 4:
                self.x += VELOCITY / math.sqrt(2)  # 오른쪽 아래
                self.y += VELOCITY / math.sqrt(2)
            elif action == 5:
                self.x -= VELOCITY / math.sqrt(2)  # 오른쪽 위
                self.y += VELOCITY / math.sqrt(2)
            elif action == 6:
                self.x += VELOCITY / math.sqrt(2)  # 왼쪽 아래
                self.y -= VELOCITY / math.sqrt(2)
            elif action == 7:
                self.x -= VELOCITY / math.sqrt(2)  # 왼쪽 위
                self.y -= VELOCITY / math.sqrt(2)
            elif action == 8:
                self.z = H_MAX
                self.counter = 0
            elif action == 9:
                self.counter += 1
                empty_mask = self.seat == -1
                empty_idx = [i for i, x in enumerate(empty_mask) if x]
                for i in range(len(empty_idx)):
                    idx = empty_idx[i]
                    for bomb in bombList:
                        if (
                            (bomb.connect["isArrive"] == 0)
                            and (bomb.connect["isOnboard"] == 0)
                            and (bomb.Departure == self.site)
                        ):
                            self.onboard_process(bomb, idx)
                            break
            elif action >= 10:
                self.site = action - 10
                self.numidx[self.site] += 1
                self.x, self.y, self.z = (
                    siteList[self.site].x,
                    siteList[self.site].y,
                    H_MIN,
                )
                seat_mask = self.seat != -1
                seat_idx = [i for i, x in enumerate(seat_mask) if x]
                for i in range(len(seat_idx)):
                    idx = seat_idx[i]
                    id = self.seat[idx]
                    if bombList[id].Arrival == siteList[self.site].id:
                        self.arrive_process(bombList, idx, id, siteList)
                empty_mask = self.seat == -1
                empty_idx = [i for i, x in enumerate(empty_mask) if x]
                for i in range(len(empty_idx)):
                    idx = empty_idx[i]
                    for bomb in bombList:
                        if (
                            (bomb.connect["isArrive"] == 0)
                            and (bomb.connect["isOnboard"] == 0)
                            and (bomb.Departure == self.site)
                        ):
                            self.onboard_process(bomb, idx)
                            break

        # state space 맞추어주기.
        self.x = self.clamp(self.x, -GRID, GRID)
        self.y = self.clamp(self.y, -GRID, GRID)
        self.batteryConsump(action)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Utility:

    def __init__(self):
        self.T_service_rate = 0
        self.T_onboard_rate = 0
        self.T_service_bomb = 0
        self.T_onboard_bomb = 0
        self.grid = 60
        self.scale = self.grid / GRID
        self.I_reward = np.zeros(N_COMMAGENT + N_DQNAGENT)
        self.I_battery = np.zeros(N_COMMAGENT + N_DQNAGENT)
        self.I_onboard_bomb = np.zeros(N_COMMAGENT + N_DQNAGENT)
        self.I_service_bomb = np.zeros(N_COMMAGENT + N_DQNAGENT)
        self.I_onboard_distance = np.zeros(N_COMMAGENT + N_DQNAGENT)
        self.BOMBARD = np.zeros(N_COMMAGENT + N_DQNAGENT + 1)
        self.num_Takeoff = np.zeros([N_SITE])

    def get_utils_info(self):
        return self.T_service_rate, self.I_battery

    def _calculate_support(self, bombList, agentList, siteList):
        for i in range(N_BOMB):
            droneId, isOnboard, isArrive = bombList[i].is_support()
            self.T_service_bomb += isArrive
            self.T_onboard_bomb += isOnboard
            if droneId != -1:
                self.I_service_bomb[droneId] += isArrive
                self.I_onboard_bomb[droneId] += isOnboard
        for j, agent in enumerate(agentList):
            seat_mask = agent.seat != -1
            seat_idx = [a for a, b in enumerate(seat_mask) if b]
            total_distance = 0
            for i in range(len(seat_idx)):
                idx = seat_idx[i]
                id = agent.seat[idx]
                x_diff = agent.x - siteList[bombList[id].Arrival].x
                y_diff = agent.y - siteList[bombList[id].Arrival].y
                total_distance += math.sqrt((x_diff) ** 2 + (y_diff) ** 2)
            if len(seat_idx) == 0:
                self.I_onboard_distance[j] = total_distance / (GRID)
            else:
                self.I_onboard_distance[j] = total_distance / (len(seat_idx) * GRID)

    def _calculate_energy_consumption(self, agentList):
        for drone in agentList:
            self.I_battery[drone.id] = drone.batteryRemain

    def _calculate_indiv_reward(self, agentList, siteList):
        for i, drone in enumerate(agentList):
            takeoff_mask = agentList[i].numidx > 0
            takeoff_idx = [m for m, n in enumerate(takeoff_mask) if n]
            self.I_reward[i] = (
                agentList[i].takeoff
                + len(takeoff_idx)
                - self.I_onboard_distance[i]
                + self.I_battery[i] / drone.maxbattery
            )  # type 1
        return self.I_reward

    def _calculate_common_reward(self, bombList, agentList, siteList):
        for i, agent in enumerate(agentList):
            self.num_Takeoff += agent.numidx
        takeoff_mask = self.num_Takeoff > 0
        takeoff_idx = [i for i, x in enumerate(takeoff_mask) if x]
        Common_Reward = np.full(
            (N_COMMAGENT + N_DQNAGENT), self.T_service_bomb / (N_DQNAGENT + N_COMMAGENT)
        )
        return Common_Reward

    def calculate_reward(self, bombList, agentList, siteList):
        self.__init__()
        self._calculate_support(bombList, agentList, siteList)
        self._calculate_energy_consumption(agentList)
        self.BOMBARD[0] = self.T_service_bomb
        self.BOMBARD[1 : (N_COMMAGENT + N_DQNAGENT + 1)] = np.copy(self.I_service_bomb)
        Common_Reward = self._calculate_common_reward(bombList, agentList, siteList)
        Individual_Reward = self._calculate_indiv_reward(agentList, siteList)
        Reward = np.array(Individual_Reward) + Common_Reward
        return Reward / 100.0


class Environment:

    def __init__(self):
        # [1] Environment
        self.EPLEN = EP_LEN
        self.numSite = N_SITE
        self.numBomb = N_BOMB
        self.numCommAgent = N_COMMAGENT
        self.numDQNAgent = N_DQNAGENT
        self.numAgent = self.numCommAgent + self.numDQNAgent
        self.GridRadius = GRID
        self.hMax = H_MAX
        self.hMin = H_MIN
        self.t = 0
        self.numTakeoff = np.zeros([self.numAgent, N_SITE])
        # [2] Environment --> RLAgent
        self.o = None
        self.o_prime = None
        # [3] Environment Initialization
        self.Initialize()

    def reset(self):
        self.Initialize()

    def get_available_action(self):
        available_action = []
        for i in range(self.numAgent):
            available_action.append(
                self.agentList[i]._avail_action(self.bombList, self.siteList)
            )
        return np.array(available_action)

    def Initialize(self):
        siteList = self._initSite()
        self._initBomb()
        self._initAgent(siteList)
        self.utility = Utility()
        self.common_obs = self.getCommonObs()
        # self.common_obs_prime = np.copy(self.common_obs)
        self.o = self.getagentObs(self.t)
        self.o_prime = np.copy(self.o)
        self.ava_prime = self.get_available_action()

    def get_info(self):
        info = dict()
        info["episode_limit"] = self.EPLEN
        info["n_Comm_agents"] = self.numCommAgent
        info["n_DQN_agents"] = self.numDQNAgent
        info["n_agents"] = self.numCommAgent + self.numDQNAgent
        info["n_actions"] = 15
        info["obs_dim"] = self.o.shape[-1]
        return info

    def _initSite(self):
        self.siteList = []
        for i in range(self.numSite):
            if i == 0:
                x, y = 0, 0  # [m]
                self.siteList.append(Site(id=i, x=x, y=y, z=H_MIN))
            elif i == 1:
                x, y = -26192, -17445
                self.siteList.append(Site(id=i, x=x, y=y, z=H_MIN))
            elif i == 2:
                x, y = 19937, 22918
                self.siteList.append(Site(id=i, x=x, y=y, z=H_MIN))
            elif i == 3:
                x, y = 21794, -13731
                self.siteList.append(Site(id=i, x=x, y=y, z=H_MIN))
            elif i == 4:
                x, y = 17396, -7183
                self.siteList.append(Site(id=i, x=x, y=y, z=H_MIN))
        return self.siteList

    def _initBomb(self):
        self.bombList = []
        for i in range(self.numBomb):
            Departure = np.random.randint(0, self.numSite)
            Arrival = np.random.randint(0, self.numSite)
            while Departure == Arrival:
                Departure = np.random.randint(0, self.numSite)
                Arrival = np.random.randint(0, self.numSite)
            self.bombList.append(Bomb(id=i, Departure=Departure, Arrival=Arrival))

    def _initAgent(self, siteList):
        self.agentList = []
        marker = np.zeros([self.numSite], dtype=int)
        for i in range(self.numAgent):
            idx = np.random.randint(0, self.numSite)
            while marker[idx] > 1:  # 0, 1
                idx = np.random.randint(0, self.numSite)
            marker[idx] += 1
            self.agentList.append(
                Drone(
                    id=i,
                    x=siteList[idx].x,
                    y=siteList[idx].y,
                    z=siteList[idx].z,
                    site=siteList[idx].id,
                )
            )

    def plot(self):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        DX, DY, DR = [], [], []
        AX, AY, AR = [], [], []
        CX, CY, CR = [], [], []
        OX, OY = [], []
        for i in range(self.numSite):
            DX.append(self.siteList[i].x)
            DY.append(self.siteList[i].y)
        for i in range(N_COMMAGENT):
            CX.append(self.agentList[i].x)
            CY.append(self.agentList[i].y)
        for i in range(N_COMMAGENT, self.numAgent):
            AX.append(self.agentList[i].x)
            AY.append(self.agentList[i].y)
        if len(DX):
            ax.scatter(DX, DY, s=90, c="k", marker="o")
        if len(CX):
            ax.scatter(CX, CY, s=90, c="magenta", marker="*")
        if len(AX):
            ax.scatter(AX, AY, s=90, c="dodgerblue", marker="*")
        if len(OX):
            ax.scatter(OX, OY, s=90, c="orangered", marker="X")
        ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=1)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.xlim([-self.GridRadius, self.GridRadius])
        plt.ylim([-self.GridRadius, self.GridRadius])
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        plt.close()
        image = ToTensor()(image).unsqueeze(0)[0]
        return image

    def getCommonObs(self):
        # [1] 요소에 대한 절대 위치
        # agent_pos = np.array([[self.agentList[k].x, self.agentList[k].y] for k in range(self.numAgent)]).flatten()
        site_pos = np.array(
            [[self.siteList[k].x, self.siteList[k].y] for k in range(self.numSite)]
        ).flatten()
        Position = np.hstack([site_pos])
        ## Normalize
        Position = Position / self.GridRadius
        # [3] Utility 정보
        Util = np.array([self.utility.T_service_rate, self.utility.T_onboard_rate])
        Util = np.append(
            Util,
            np.array(
                [
                    self.utility.I_service_bomb,
                    self.utility.I_onboard_bomb,
                    self.utility.I_onboard_distance,
                ]
            ).flatten(),
        )
        # 모든 Observation을 concatenate
        # CommonObs     = np.hstack([Position, Connection, Util])
        CommonObs = np.hstack([Position, Util])
        return CommonObs

    def getagentObs(self, t):
        Obs = []
        obs_common = self.common_obs
        for i in range(self.numAgent):
            # o_partial = []
            # [1] 자신의 정보
            # agent = self.agentList[i] # 현재 주체가 되는 Agent의 index == i
            o_pos = []
            o_pos.append(self.agentList[i].x / self.GridRadius)
            o_pos.append(self.agentList[i].y / self.GridRadius)
            o_pos.append(self.agentList[i].z / self.GridRadius)
            o_pos.append(self.agentList[i].batteryRemain / self.agentList[i].maxbattery)
            # o_pos.append(self.agentList[i].batteryRemain)
            # [2] 다른 Agent의 위치 정보
            o_pos_agent = []
            for k in range(self.numAgent):
                x_diff = self.agentList[i].x - self.agentList[k].x
                y_diff = self.agentList[i].y - self.agentList[k].y
                diff = math.sqrt((x_diff) ** 2 + (y_diff) ** 2)
                if diff <= OBSERVABLE:
                    o_pos_agent.append(self.agentList[k].x / self.GridRadius)
                    o_pos_agent.append(self.agentList[k].y / self.GridRadius)
                    # o_pos_agent.append(self.agentList[k].z / self.GridRadius)
                    o_pos_agent.append(diff / self.GridRadius)
                else:
                    o_pos_agent.append(-1)
                    o_pos_agent.append(-1)
                    # o_pos_agent.append(-1)
                    o_pos_agent.append(-1)
            # [3] Onloaded Passenger의 정보
            o_seat = []
            seat_mask = self.agentList[i].seat != -1
            seat_idx = [j for j, x in enumerate(seat_mask) if x]
            seat_counter = 0
            for k in range(Max_num_bomb):
                if seat_mask[k]:
                    idx = seat_idx[seat_counter]
                    seat_counter += 1
                    id = self.agentList[i].seat[idx]
                    o_seat.append(self.bombList[id].Arrival / N_SITE)
                    o_seat.append(self.bombList[id].Departure / N_SITE)
                else:
                    # o_seat.append(-1)
                    o_seat.append(-1)
                    o_seat.append(-1)
            o_pos_site = []
            for k in range(self.numSite):
                x_diff = self.agentList[i].x - self.siteList[k].x
                y_diff = self.agentList[i].y - self.siteList[k].y
                diff = math.sqrt((x_diff) ** 2 + (y_diff) ** 2)
                if diff <= OBSERVABLE:
                    o_pos_site.append(self.siteList[k].x / self.GridRadius)
                    o_pos_site.append(self.siteList[k].y / self.GridRadius)
                    o_pos_site.append(diff / self.GridRadius)
                else:
                    o_pos_site.append(-1)
                    o_pos_site.append(-1)
                    # o_pos_site.append(-1)
                    o_pos_site.append(-1)
            o_pos = np.array(o_pos)
            o_pos_agent = np.array(o_pos_agent)
            o_pos_site = np.array(o_pos_site)
            o_seat = np.array(o_seat)
            o_idx = np.zeros(self.numAgent)
            o_idx[self.agentList[i].id] += 1
            o_partial = np.hstack(
                [o_pos, o_pos_agent, o_pos_site, o_seat]
            )  # 모든 정보 Concatenation
            # o = np.hstack([o_idx, obs_common, o_partial])
            # o = np.hstack([t/EP_LEN, o_idx, o_partial])
            o = np.hstack([o_idx, o_partial])
            Obs.append(np.copy(o))
        Obs = np.array(Obs)
        return Obs

    def step(self, actions, time, epoch, today, info):
        self.t = time
        self.o = np.copy(self.o_prime)
        for i in range(self.numAgent):
            self.agentList[i].transition(actions[i], self.bombList, self.siteList)
        # self.getCommonObs()
        self.o_prime = self.getagentObs(time)
        rewards = self.utility.calculate_reward(
            self.bombList, self.agentList, self.siteList
        )
        if time == 0:
            self.x_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.y_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.z_COMM = np.zeros([EP_LEN, N_COMMAGENT])
            self.x_DNN = np.zeros([EP_LEN, N_DQNAGENT])
            self.y_DNN = np.zeros([EP_LEN, N_DQNAGENT])
            self.z_DNN = np.zeros([EP_LEN, N_DQNAGENT])
        if time == EP_LEN:
            for i, agent in enumerate(self.agentList):
                self.numTakeoff[i] = agent.numidx
        self.ava_prime = self.get_available_action()
        if time < EP_LEN:
            for i in range(N_COMMAGENT):
                self.x_COMM[time, i] = self.agentList[i].x
                self.y_COMM[time, i] = self.agentList[i].y
                self.z_COMM[time, i] = self.agentList[i].z
            for i in range(N_COMMAGENT, self.numAgent):
                self.x_DNN[time, i - N_COMMAGENT] = self.agentList[i].x
                self.y_DNN[time, i - N_COMMAGENT] = self.agentList[i].y
                self.z_DNN[time, i - N_COMMAGENT] = self.agentList[i].z
        return (
            self.o,
            rewards,
            self.o_prime,
            self.ava_prime,
            self.numTakeoff,
            self.x_COMM,
            self.y_COMM,
            self.x_DNN,
            self.y_DNN,
        )
