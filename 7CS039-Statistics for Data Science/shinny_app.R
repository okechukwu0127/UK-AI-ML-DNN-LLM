library(shiny)

# Define UI for app 
ui <- fluidPage(
  
  # App title ----
  titlePanel("Exploring the Bodyfat data!"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      sliderInput(inputId = "bins",
                  label = "Number of bins:",
                  min = 1,
                  max = 50,
                  value = 30)
      
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Histogram ----
      plotOutput(outputId = "distPlot")
      
    )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {

  output$distPlot <- renderPlot({
    Bodyfat <- read.csv("/Users/oeze/Documents/wlv/7CS039/Bodyfat.csv")
    x    <- Bodyfat$Weight
    bins <- seq(min(x), max(x), length.out = input$bins + 1)
    
    hist(x, breaks = bins, col = "#75AADB", border = "white",
         xlab = "Whatever the x is",
         main = "Histogram of Whatever this is")
    
  })
  
  
}

# Create Shiny app ----
shinyApp(ui = ui, server = server)
