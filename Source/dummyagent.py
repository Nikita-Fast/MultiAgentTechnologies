import asyncio

import numpy as np
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template
from adj_matrix import ADJ_MATRIX
import json

AGENT_STARTED = {}


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

    @property
    def local_mean(self):
        return np.mean(self.array)

    def _set_neighbours(self):
        # +-1 т.к. id агентов от 1 до N, а индексы матрицы смежности от 0 до N-1
        self.neighbours = (ADJ_MATRIX[self.id - 1]).nonzero()[0] + 1

    async def setup(self):
        print(f"[{self.id}] Agent started!")

        b1 = self.SendNumbersBehav()
        t1 = Template()
        t1.set_metadata("performative", "request_nums")
        self.add_behaviour(b1, t1)

        b2 = self.RequestReceiveNumbersBehav()
        t2 = Template()
        t2.set_metadata("performative", "response_nums")
        self.add_behaviour(b2, t2)

        self.request_receive_behav = b2

    @staticmethod
    def id_to_jid(id: int):
        return f"nikita_agent{id}@07f.de"

    class RequestReceiveNumbersBehav(CyclicBehaviour):
        # На каждой итерации агент спрашивает у соседей неизвестные ему числа.
        # Для этого агент отправляет сообщение вида (agent_id, unknown_nums_idx).
        # После этого агент ожидает ответ от каждого из своих соседей.
        # Итерации этого поведения продолжаются пока агент не узнает все числа.

        async def on_start(self):
            self.agent.iter_cnt = 0
            self.agent.request_cnt = 0
            AGENT_STARTED[self.agent.id - 1] = 1
            agents_number = len(self.agent.array)
            while len(AGENT_STARTED.keys()) < agents_number:
                await asyncio.sleep(0)
            print(f"[{self.agent.id}] Starting RequestReceiveNumbersBehav behaviour . . .")

        async def run(self):
            agent: MyAgent = self.agent
            self.agent.iter_cnt += 1

            unknown_nums_idx = [i for i, v in enumerate(self.agent.array) if v is None]
            for neighbour_id in agent.neighbours:
                await self._send_msg(dst_id=neighbour_id, payload=(agent.id, unknown_nums_idx), mark="request_nums")

            cnt = 0
            for neighbour_id in agent.neighbours:
                msg = await self.receive(timeout=5)
                if msg:
                    cnt += 1
                    print(f"[{agent.id}] Ответ от агента-соседа {neighbour_id}: {msg.body}")
                    j, _dict = json.loads(msg.body)
                    self._update_local_array(_dict)

            if cnt != len(agent.neighbours):
                raise RuntimeError(f"[{self.agent.id}] Не все соседи прислали свои числа!")

            if np.sum(np.array(self.agent.array) == None) == 0:
                print(f"Agent [{self.agent.id}]: все числа собраны")
                self.kill(exit_code=228)

        async def _send_msg(self, dst_id, payload, mark):
            neighbour_jid = MyAgent.id_to_jid(dst_id)
            msg = Message(to=neighbour_jid)
            msg.set_metadata("performative", mark)
            msg.body = json.dumps(payload)

            print(f"[{self.agent.id}] -> [{dst_id}] data={msg.body}")
            await self.send(msg)
            self.agent.request_cnt += 1

        def _update_local_array(self, _dict: dict):
            for k, v in _dict.items():
                if v is None:
                    raise RuntimeError(f"[{self.agent.id}]: Передали вместо числа None")
                self.agent.array[int(k)] = v

    class SendNumbersBehav(OneShotBehaviour):
        # Агент получает от соседа запрос значений неизвестных чисел.
        # Агент проверяет какие из этих чисел он знает.
        # Агент отправляет известные числа соседу.

        async def _send_msg(self, dst_id, payload, mark):
            neighbour_jid = MyAgent.id_to_jid(dst_id)
            msg = Message(to=neighbour_jid)
            msg.set_metadata("performative", mark)
            msg.body = json.dumps(payload)

            print(f"[{self.agent.id}] -> [{dst_id}] data={msg.body}")
            await self.send(msg)
            self.agent.response_cnt += 1

        async def on_start(self) -> None:
            self.agent.response_cnt = 0

        async def run(self) -> None:
            agent: MyAgent = self.agent

            while True:
                while self.mailbox_size() == 0:
                    await asyncio.sleep(0)
                print(f"[{self.agent.id}] SendNumbersBehav behaviour . . .")

                msg = await self.receive(timeout=5)
                request_agent_id, unknown_nums_idx = json.loads(msg.body)

                numbers = {i: self.agent.array[i] for i in unknown_nums_idx if self.agent.array[i] is not None}
                await self._send_msg(request_agent_id, (self.agent.id, numbers), "response_nums")


async def main():
    if not np.all(np.transpose(ADJ_MATRIX) == ADJ_MATRIX):
        raise RuntimeError("Матрицы смежности должна быть симметрична!")

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

    while not all(agent.request_receive_behav.is_killed() for agent in agents):
        try:
            await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

    for agent in agents:
        if agent.request_receive_behav.exit_code != 228:
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
