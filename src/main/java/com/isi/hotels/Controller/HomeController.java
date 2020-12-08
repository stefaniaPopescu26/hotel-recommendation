package com.isi.hotels.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class HomeController {

    @GetMapping("/")
    public String index(){

        return "index";
    }

    @GetMapping("/form")
    public String form(){

        return "form";
    }

    @PostMapping("/send")
    public String send(@RequestParam("distanta")float distanta,
                       @RequestParam("anCazare")String anCazare,
                       @RequestParam("lunaCazare")String lunaCazare,
                       @RequestParam("ziCazare")String ziCazare,
                       @RequestParam("anDecazare")String anDecazare,
                       @RequestParam("lunaDecazare")String lunaDecazare,
                       @RequestParam("ziDecazare")String ziDecazare,
                       @RequestParam("numarAdulti")int numarAdulti,
                       @RequestParam("numarCopii")int numarCopii,
                       @RequestParam("numarCamere")int numarCamere){

        System.out.println("distanta = " + distanta);
        System.out.println("anCazare = " + anCazare);
        System.out.println("lunaCazare = " + lunaCazare);
        System.out.println("ziCazare = " + ziCazare);
        System.out.println("anDecazare = " + anDecazare);
        System.out.println("lunaDecazare = " + lunaDecazare);
        System.out.println("ziDecazare = " + ziDecazare);
        System.out.println("numarAdulti = " + numarAdulti);
        System.out.println("numarCopii = " + numarCopii);
        System.out.println("numarCamere = " + numarCamere);

        return "redirect:/result";
    }

    @GetMapping("/result")
    public String result(){

        return "result";
    }

}
