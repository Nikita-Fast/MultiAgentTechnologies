import asyncio
import time

import numpy as np
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from adj_matrix import ADJ_MATRIX
import json

AGENT_STARTED = {}


class MyAgent(Agent):
    def __init__(self, id: int, number: float | int, N: int):
        self.neighbours: list[int] = None
        self.id = id
        self.number = number
        self.array = [None] * N
        self.N = N
        self.local_mean = None
        jid = MyAgent.id_to_jid(id)
        super().__init__(jid, "Nikitafast1404")

    def _set_neighbours(self):
        # +-1 т.к. id агентов от 1 до N, а индексы матрицы смежности от 0 до N-1
        self.neighbours = (ADJ_MATRIX[self.id - 1]).nonzero()[0] + 1

    def _init(self):
        # +-1 т.к. id агентов от 1 до N, а индексы массива от 0 до N-1
        self.array[self.id - 1] = self.number
        self._set_neighbours()
        print(f"[{self.id}] neighbours={self.neighbours}")

    async def setup(self):
        print(f"[{self.id}] Agent started!")
        self._init()

        behav = self.SendReceiveNumbersBehav()
        self.behav = behav
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(behav, template)

    @staticmethod
    def id_to_jid(id: int):
        agent_id_to_jid = {
            1: "nikita_agent1@07f.de",
            2: "nikita_agent2@07f.de",
            3: "nikita_agent3@07f.de",
            4: "nikita_agent4@07f.de",
            5: "nikita_agent5@07f.de"
        }
        return agent_id_to_jid[id]

    class SendReceiveNumbersBehav(CyclicBehaviour):
        def __init__(self):
            super().__init__()

        async def on_start(self):
            self.agent.iter_cnt = 0
            self.agent.msg_sent_cnt = 0
            self.agent.msg_recv_cnt = 0
            AGENT_STARTED[self.agent.id - 1] = 1
            while len(AGENT_STARTED.keys()) < self.agent.N:
                await asyncio.sleep(0)
            print(f"[{self.agent.id}] Starting SendReceiveNumbersBehav behaviour . . .")

        async def run(self):
            # пока в массиве есть хотя бы один None
            if np.sum(np.array(self.agent.array) == None) > 0:
                self.agent.iter_cnt += 1
                await self._send_numbers()
                await self._receive_numbers()
            else:
                self.kill(exit_code=228)
                self.agent.local_mean = np.mean(self.agent.array)
                print(f"[Agent{self.agent.id}] Done in {self.agent.iter_cnt} iterations! Msg tx {self.agent.msg_sent_cnt}, Msg rx {self.agent.msg_recv_cnt}")

        async def _send_numbers(self):
            agent: MyAgent = self.agent

            for neighbour_id in agent.neighbours:
                self.agent.msg_sent_cnt += 1
                await self._send_msg(dst_id=neighbour_id, payload=(agent.id, agent.array))

        async def _receive_numbers(self):
            agent: MyAgent = self.agent

            cnt = 0
            for _ in agent.neighbours:
                msg = await self.receive(timeout=5)
                self.agent.msg_recv_cnt += 1
                if msg:
                    cnt += 1
                    print(f"[{agent.id}] Ответ от соседа: {msg.body}")
                    j, array = json.loads(msg.body)
                    self._update_local_array(array)

            if cnt != len(agent.neighbours):
                raise RuntimeError("Не все соседи прислали свои числа!")

        async def _send_msg(self, dst_id, payload):
            neighbour_jid = MyAgent.id_to_jid(dst_id)
            msg = Message(to=neighbour_jid)
            msg.set_metadata("performative", "inform")
            msg.body = json.dumps(payload)

            print(f"[{self.agent.id}] -> [{dst_id}] data={msg.body}")
            await self.send(msg)

        def _update_local_array(self, neighbour_array):
            for (i, N_i) in enumerate(neighbour_array):
                if (self.agent.array[i] is None) and (N_i is not None):
                    self.agent.array[i] = N_i


async def main():
    N = 5
    numbers = [4.5, 2.1, -7.3, 1.2, 3.9]

    agent1 = MyAgent(1, numbers[0], N)
    agent2 = MyAgent(2, numbers[1], N)
    agent3 = MyAgent(3, numbers[2], N)
    agent4 = MyAgent(4, numbers[3], N)
    agent5 = MyAgent(5, numbers[4], N)

    agents = [agent1, agent2, agent3, agent4, agent5]

    for agent in agents:
        await agent.start()

    agent1.web.start(hostname="127.0.0.1", port="10000")
    agent2.web.start(hostname="127.0.0.1", port="10001")

    while not agent1.behav.is_killed():
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break

    for agent in agents:
        if agent.behav.exit_code != 228:
            raise RuntimeError(f"Agent{agent.id} завершился с неправильным статус кодом")

    for agent in agents:
        await agent.stop()

    print("-----------------------------")
    for agent in agents:
        print(f"[Agent{agent.id}] answer = {agent.local_mean}")

    print(f"Control answer = {np.mean(numbers)}")


if __name__ == "__main__":
    spade.run(main())


