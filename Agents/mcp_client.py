import asyncio
from fastmcp import Client
import weather_server

async def main():
    # Conect to MCP server using la instance of the server
    async with Client(weather_server.mcp) as mcp_client:
        # Get list of tools
        tools = await mcp_client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {getattr(tool, 'description', '')}")

        # Test call to get_weather
        response = await mcp_client.call_tool("get_weather", {"city": "Berlin"})
        print(f"\nTemperature in Berlin: {response.data}")

        # Test call to set_weather
        response2 = await mcp_client.call_tool("set_weather", {"city": "Paris", "temp": 28.3})
        print(f"\nAnswer set_weather: {response2.data}")

        response3 = await mcp_client.call_tool("get_weather", {"city": "Paris"})
        print(f"\nTemperature in Paris: {response3.data}")

if __name__ == "__main__":
    asyncio.run(main())
