import asyncio
from data_clean_env.client import DataCleanEnv
from data_clean_env.models import DataCleanAction

async def test_live():
    print('Connecting to live Hugging Face Space...')
    # Use HTTP client mode pointing to the space
    env = DataCleanEnv(base_url='https://mg0718-data-clean-env.hf.space')
    
    async with env:
        print('Resetting environment...')
        res = await env.reset(task_name='fix_missing_values')
        print('> Successfully started Task:', res.observation.task_name)
        
        print('Sending inspect command...')
        res2 = await env.step(DataCleanAction(command='inspect'))
        print('> Issues to fix:', len(res2.observation.current_issues))
        
        print('Submitting...')
        res3 = await env.step(DataCleanAction(command='submit'))
        print('> Final Score:', res3.observation.score_so_far)
        print('SUCCESS! The Hugging Face server is fully responsive.')

if __name__ == "__main__":
    asyncio.run(test_live())
