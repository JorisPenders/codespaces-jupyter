import aiohttp
import asyncio
import time

start_time = time.time()


async def get_pokemon(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.text()
        # print(pokemon['meta']['general_statistics']['nr_plays'])
        return pokemon[:10]


async def main():

    async with aiohttp.ClientSession() as session:

        tasks = []
        for number in range(1, 300):
            url = f'http://musicburst.jorispenders.nl/'
            tasks.append(asyncio.ensure_future(get_pokemon(session, url)))

        original_pokemon = await asyncio.gather(*tasks)


asyncio.run(main())
print("--- %s seconds ---" % (time.time() - start_time))