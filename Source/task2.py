import asyncio
import random
import sys
import matplotlib

DISABLE_ITER_MATPLOTLIB = False
DISABLE_PLOTTING = False

if sys.version_info.minor <= 10:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import numpy as np
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from adj_matrix import ADJ_MATRIX
import json

agent_values = {}
ITER_NUM = 100

phase1_event = asyncio.Event()
PHASE1_STARTED = {}
async def sync_phase1_start():
    while True:
        if len(PHASE1_STARTED.keys()) == 5:
            PHASE1_STARTED.clear()
            phase2_event.clear()
            phase1_event.set()
        await asyncio.sleep(0)

phase2_event = asyncio.Event()
PHASE2_STARTED = {}
async def sync_phase2_start():
    while True:
        if len(PHASE2_STARTED.keys()) == 5:
            PHASE2_STARTED.clear()
            phase1_event.clear()
            phase2_event.set()
        await asyncio.sleep(0)


async def sync_phase12():
    while not sync_phase12.done:
        if len(PHASE1_STARTED.keys()) == 5:
            if not DISABLE_ITER_MATPLOTLIB:
                plt.ion()
                _plot()
            PHASE1_STARTED.clear()
            phase2_event.clear()
            phase1_event.set()
        if len(PHASE2_STARTED.keys()) == 5:
            PHASE2_STARTED.clear()
            phase1_event.clear()
            phase2_event.set()
        await asyncio.sleep(0)

sync_phase12.done = False

ALPHA = 1/30


def _plot():
    if not DISABLE_PLOTTING:
        data = list(zip(*agent_values.values()))

        plt.clf()
        plt.axhline(y=_plot._ideal_mean, color='r', linestyle='-')
        plt.plot(data)
        plt.draw()
        plt.pause(0.001)



class MyAgent(Agent):
    def __init__(self, id: int, number: float | int, N: int):
        self.neighbours: list[int] = None
        self.id = id
        self.phase = 1
        self.iteration = 0
        self._init(id, number, N)
        jid = MyAgent.id_to_jid(id)

        super().__init__(jid, "Nikitafast1404")

    def _init(self, id, number, N):
        # +-1 т.к. id агентов от 1 до N, а индексы массива от 0 до N-1
        self.array = np.array([number] * N, dtype=float)
        self.array[self.id - 1] = number
        self._set_neighbours()
        print(f"[{self.id}] neighbours={self.neighbours}")

    def get_number(self, agent_id):
        return self.array[agent_id - 1]

    def set_number(self, agent_id, val):
        self.array[agent_id - 1] = val

    @property
    def local_mean(self):
        return np.mean(self.array)

    def _set_neighbours(self):
        # +-1 т.к. id агентов от 1 до N, а индексы матрицы смежности от 0 до N-1
        self.neighbours = (ADJ_MATRIX[self.id - 1]).nonzero()[0] + 1

    def get_connection_probability(self, dst_agent_id: int):
        return ADJ_MATRIX[self.id - 1, dst_agent_id - 1]

    async def setup(self):
        print(f"[{self.id}] Agent started!")

        b2 = self.ComboBehav()
        t2 = Template()
        t2.set_metadata("performative", "MARK_SEND")
        self.add_behaviour(b2, t2)

        self.RecvBehav = b2

    @staticmethod
    def id_to_jid(id: int):
        return f"nikita_agent{id}@07f.de"


    class ComboBehav(CyclicBehaviour):
        async def on_start(self) -> None:
            agent_values[self.agent.id] = []

        async def _send_msg(self, dst_id, payload, mark):
            neighbour_jid = MyAgent.id_to_jid(dst_id)
            msg = Message(to=neighbour_jid)
            msg.set_metadata("performative", mark)
            msg.body = json.dumps(payload)
            await self.send(msg)

        async def send_to_neighbours(self):
            for neighbour_id in self.agent.neighbours:
                neighbour_id = int(neighbour_id)
                p = self.agent.get_connection_probability(neighbour_id)
                has_connection = random.choices([1, 0], weights=[p, 1 - p])[0]
                # has_connection = True
                if has_connection:
                    value = self.agent.get_number(self.agent.id)
                    noise = np.random.normal(0, 0.1, 1)[0]
                    await self._send_msg(dst_id=neighbour_id, payload=(self.agent.id, value + noise), mark="MARK_SEND")
                else:
                    pass
                    print(f"[{self.agent.id}] -> [{neighbour_id}] no connection")

        async def recv_from_neighbours(self):
            n_values = []
            for _ in self.agent.neighbours:
                msg = await self.receive()
                if msg:
                    j, x_j = json.loads(msg.body)
                    self.agent.set_number(j, x_j)
                    n_values.append(x_j)
                else:
                    print(f"[{self.agent.id}] missed value")
            if len(n_values) < len(self.agent.neighbours):
                pass
                print(f"recv {len(n_values)} / {len(self.agent.neighbours)}")

            i = self.agent.id
            x_i = self.agent.get_number(i)
            _change = 0
            for j in self.agent.neighbours:
                x_j = self.agent.get_number(j)
                _change += ALPHA * (x_j - x_i)
            x_i_new = x_i + _change
            # if self.agent.id == 5:
            #     print("agent5:", self.agent.array, x_i, _change, x_i_new)
            self.agent.set_number(i, x_i_new)
            agent_values[i].append(x_i_new)

        async def run(self):
            match self.agent.phase:
                case 1:
                    PHASE1_STARTED[self.agent.id] = 1
                    await phase1_event.wait()

                    if self.agent.iteration < ITER_NUM:
                        await self.send_to_neighbours()
                        self.agent.phase = 2
                    else:
                        self.agent.phase = 3
                case 2:
                    PHASE2_STARTED[self.agent.id] = 1
                    await phase2_event.wait()

                    await self.recv_from_neighbours()
                    self.agent.phase = 1
                    self.agent.iteration += 1
                case _:
                    print(f"agent {self.agent.id} res:", self.agent.get_number(self.agent.id))
                    sync_phase12.done = True
                    self.kill(228)



async def main():
    N = 5
    # numbers = [1, 20, 40, 60, 105]
    # numbers = [8,22,15,5,0]
    numbers = [-7592, 1465, 9977, -37289, 8754]

    agent1 = MyAgent(1, numbers[0], N)
    agent2 = MyAgent(2, numbers[1], N)
    agent3 = MyAgent(3, numbers[2], N)
    agent4 = MyAgent(4, numbers[3], N)
    agent5 = MyAgent(5, numbers[4], N)

    print(F"IDEAL_MEAN = {np.mean(numbers)}")

    agents = [agent1, agent2, agent3, agent4, agent5]

    _plot._ideal_mean = np.mean(numbers)

    for agent in agents:
        await agent.start()

    await sync_phase12()

    while not all(agent.RecvBehav.is_killed() for agent in agents):
        try:
            await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

    for agent in agents:
        if agent.RecvBehav.exit_code != 228:
            raise RuntimeError(f"Agent{agent.id} завершился с неправильным статус кодом")

    for agent in agents:
        await agent.stop()

    data = list(zip(*agent_values.values()))
    _std = np.std(data[-1])
    _dist = np.abs(data[-1][0] - _plot._ideal_mean)
    print("            std(agents values) =", _std)
    print("abs( agent1_val - ideal_mean ) =", _dist)

    plt.ioff()
    _plot()
    plt.show()



if __name__ == "__main__":
    spade.run(main())