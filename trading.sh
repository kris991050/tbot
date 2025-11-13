#!/bin/sh

arg=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5
arg6=$6

# if [ "$arg" = "docker" ]
# then
#     echo "Opening Docker..."
#     # if (!docker info > /dev/null 2>&1)
#     if ((docker info | grep ERROR) > /dev/null 2>&1)
#     then
#         open -a Docker
#         echo "Waiting 25 sec..."
#         dockerAlreadyStarted=false
#         sleep 25
#     else
#         echo "Docker already open"
#         dockerAlreadyStarted=true
#     fi

#     cd /Volumes/untitled/Trading/trading_app
#     echo "Starting Docker trading script..."
#     # docker run -it --rm --network="host" -v /Volumes/untitled/Invest/Trading/trading_app:/app --name python_trading python_trading_img $arg
#     docker run -it --rm -v /Volumes/untitled/Invest/Trading/trading_app:/app -p "7497:7400" --name python_trading python_trading_img

echo "Activating venv3.12 environment..."
source ~/venv3.12/bin/activate


if [ "$arg" = "quest" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/invest
    echo "Starting questrade script..."
    python questrade.py $arg2

elif [ "$arg" = "scanner" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting scanner script..."
    python scanner.py $arg2 $arg3 $arg4

elif [ "$arg" = "news" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting news script..."
    python newsScraper.py

elif [ "$arg" = "journal" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting journal script..."
    python journal.py $arg2 $arg3 $arg4 $arg5 $arg6

elif [ "$arg" = "server" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting webhookServer script..."
    sudo python webhookServer.py

elif [ "$arg" = "post" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting testPostRequest script..."
    python testPostRequest.py

elif [ "$arg" = "order" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting orders script..."
    python orders.py $arg2

elif [ "$arg" = "stream" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting stream script..."
    python stream.py

elif [ "$arg" = "chart" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting chart script..."
    python chart.py

elif [ "$arg" = "station" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting station script..."
    python station.py $arg2

elif [ "$arg" = "level2" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting level2 script..."
    python level2.py $arg2 $arg3 $arg4

elif [ "$arg" = "patterns" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/utils
    echo "Starting patterns script..."
    python patterns.py $arg2 $arg3 $arg4

elif [ "$arg" = "timeNSales" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts
    echo "Starting tmeNSales script..."
    python timeNSales.py $arg2 $arg3

elif [ "$arg" = "data" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/data
    echo "Starting hist_market_data_handler script..."
    python hist_market_data_handler.py $arg2 $arg3 $arg4

elif [ "$arg" = "stats" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/stats
    echo "Starting stats script..."
    python stats.py $arg2 $arg3 $arg4

elif [ "$arg" = "stats_RSI-reversal" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/stats
    echo "Starting stats_RSI-reversal script..."
    python stats_RSI-reversal.py $arg2 $arg3

elif [ "$arg" = "liveScan" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/live
    echo "Starting live_scans_fetcher script..."
    python live_scans_fetcher.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "liveL2" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/live
    echo "Starting live_L2_fetcher script..."
    python live_L2_fetcher.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "liveOrchestrator" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/live
    echo "Starting live_orchestrator script..."
    python live_orchestrator.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "liveWorker" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/live
    echo "Starting live_worker script..."
    python live_worker.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "sr" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/indicators
    echo "Starting support_resistance script..."
    python support_resistance.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "ml" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/ml
    echo "Starting ml_trainer script..."
    python ml_trainer.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "backtest" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/backtest
    echo "Starting custom_backtest script..."
    python backtest_orchestrator.py $arg2 $arg3 $arg4 $arg5 $arg6

elif [ "$arg" = "get_stock_list" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/data
    echo "Starting stock_list_builder script..."
    python stock_list_builder.py $arg2 $arg3 $arg4 $arg5

elif [ "$arg" = "test_polygon" ]
then

    cd /Volumes/untitled/Trading/trading_app/scripts/data
    echo "Starting test_polygon script..."
    python test_polygon.py $arg2 $arg3 $arg4 $arg5

else

    cd /Volumes/untitled/Trading/trading_app
    echo "Starting trading script..."
    python trading.py

fi



# if [ $dockerAlreadyStarted = false ]
# then
#     echo "Closing Docker..."
#     killall -a Docker
# else
    #echo "Keeping Docker running..."
# fi
