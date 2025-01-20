import asyncio
import random

import numpy as np
import spade
from matplotlib import pyplot as plt
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template
from adj_matrix import ADJ_MATRIX
import json

AGENT_STARTED = {}
agent_values = {}


xuixui = dict()
send_event = asyncio.Event()
recv_event = asyncio.Event()


def _plot():
    data = list(zip(*agent_values.values()))
    print("agents vals std", np.std(data[-1]))
    print("agent 1 and ideal value diff", block_agents._ideal_mean - data[-1][0])

    plt.clf()
    plt.axhline(y=block_agents._ideal_mean, color='r', linestyle='-')
    plt.plot(data)
    plt.draw()
    plt.pause(0.001)


async def block_agents():
    plt.ion()
    while True:
        if len(xuixui.keys()) == 5:
            print("**************** ALL AGENTS DONE ITERATION ****************")
            plt.ion()
            _plot()
            xuixui.clear()

            recv_event.set()
            recv_event.clear()

            send_event.set()
            send_event.clear()
        await asyncio.sleep(0)


class MyAgent(Agent):
    def __init__(self, id: int, number: float | int, N: int):
        self.neighbours: list[int] = None
        self.id = id
        self._init(id, number, N)
        jid = MyAgent.id_to_jid(id)

        super().__init__(jid, "Nikitafast1404")

    def _init(self, id, number, N):
        # +-1 т.к. id агентов от 1 до N, а индексы массива от 0 до N-1
        self.array = [None] * N
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

        b1 = self.RecvBehav()
        self.add_behaviour(b1)

        b2 = self.SendBehav()
        t2 = Template()
        t2.set_metadata("performative", "MARK_SEND")
        self.add_behaviour(b2, t2)

        self.RecvBehav = b2

    @staticmethod
    def id_to_jid(id: int):
        return f"nikita_agent{id}@07f.de"

    class SendBehav(CyclicBehaviour):

        async def on_start(self):
            print(f"[{self.agent.id}] start SendBehav")
            AGENT_STARTED[self.agent.id - 1] = 1
            agents_number = len(self.agent.array)
            while len(AGENT_STARTED.keys()) < agents_number:
                await asyncio.sleep(0)

        async def _send_msg(self, dst_id, payload, mark):
            neighbour_jid = MyAgent.id_to_jid(dst_id)
            msg = Message(to=neighbour_jid)
            msg.set_metadata("performative", mark)
            msg.body = json.dumps(payload)

            # print(f"[{self.agent.id}] -> [{dst_id}] data={msg.body}")
            await self.send(msg)

        async def run(self) -> None:
            # На каждой итерации отправляем всем соседям свое значение
            # Значение при передачи складывается с шумом
            # Связь с соседями может эпизодически исчезать

            agent: MyAgent = self.agent

            while True:
                print(f"[{self.agent.id}]: START ITER {self.agent.iter_cnt}")
                for neighbour_id in agent.neighbours:
                    neighbour_id = int(neighbour_id)
                    p = agent.get_connection_probability(neighbour_id)
                    has_connection = random.choices([1, 0], weights=[p, 1-p])[0]
                    # has_connection = True
                    if has_connection:
                        value = agent.get_number(agent.id)
                        noise = np.random.normal(0, 0.1, 1)[0]
                        # noise = 0
                        await self._send_msg(dst_id=neighbour_id, payload=(agent.id, value + noise), mark="MARK_SEND")
                    else:
                        pass
                        print(f"[{self.agent.id}] -> [{neighbour_id}] no connection")
                await send_event.wait()


    class RecvBehav(CyclicBehaviour):
        async def on_start(self):
            self.agent.iter_cnt = 0
            agent_values[self.agent.id] = []

        async def run(self):
            print(f"[{self.agent.id}] start RecvBehav")
            agent: MyAgent = self.agent

            # получяем числа от агентов соседей
            for _ in agent.neighbours:
                msg = await self.receive(timeout=5)
                if msg:
                    j, x_j = json.loads(msg.body)
                    print(f"[{agent.id}] Получено значение от агента соседа {j}: {x_j}")
                    agent.set_number(j, x_j)
                else:
                    print(f"[{self.agent.id}] do not get value")

            # расчитываем собственное значение, используя Local Voting Protocol
            # Если текущее значение соседа не поступило, то будет использовано предыдущее
            x_i = agent.get_number(agent.id)
            alpha = 1/10
            _sum = 0
            vals = [agent.get_number(j) for j in agent.neighbours]
            vals = [x for x in vals if x is not None]

            x_i = x_i + alpha * np.sum(np.array(vals) - x_i)
            agent.set_number(agent.id, x_i)

            agent_values[self.agent.id].append(x_i)

            print(f"[{self.agent.id}]: FINISH ITER {self.agent.iter_cnt}, CUR VAL = {agent.get_number(agent.id)}")

            xuixui[self.agent.id] = self.agent.iter_cnt

            self.agent.iter_cnt += 1
            await recv_event.wait()


async def main():
    if not np.all(np.transpose(ADJ_MATRIX) == ADJ_MATRIX):
        raise RuntimeError("Матрицы смежности должна быть симметрична!")

    N = 5
    numbers = [1000, 2000, 3321, 4000, 5000]

    agent1 = MyAgent(1, numbers[0], N)
    agent2 = MyAgent(2, numbers[1], N)
    agent3 = MyAgent(3, numbers[2], N)
    agent4 = MyAgent(4, numbers[3], N)
    agent5 = MyAgent(5, numbers[4], N)

    print(F"IDEAL_MEAN = {np.mean(numbers)}")

    agents = [agent1, agent2, agent3, agent4, agent5]

    block_agents._ideal_mean = np.mean(numbers)

    for agent in agents:
        await agent.start()

    await block_agents()


    agent1.web.start(hostname="127.0.0.1", port="10000")
    agent2.web.start(hostname="127.0.0.1", port="10001")

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

    print("-----------------------------")
    for agent in agents:
        print(
            f"[Agent{agent.id}] mean = {agent.local_mean}, iter_n={agent.iter_cnt}, request_n={agent.request_cnt}, response_n={agent.response_cnt}")

    print(f"Control mean = {np.mean(numbers)}")


if __name__ == "__main__":
    spade.run(main())
